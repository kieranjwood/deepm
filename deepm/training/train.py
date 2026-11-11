import os
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import wandb
from torch import nn

from deepm.data.dataset import MomentumDataset, unpack_torch_dataset
from deepm.models.common import LossFunction
from deepm.models.deepm import AdvancedTemporalBaseline, SpatioTemporalGraphTransformer
from deepm.models.base import DeepMomentumNetwork, DmnMode
from deepm.models.lstm import LstmBaseline, LstmSimple
from deepm.models.momentum_transformer import MomentumTransformer
from deepm.training.grad_accum import train_step_exact_chunked
from deepm.utils.logging_utils import get_logger
from deepm.utils.metrics import calmar_ratio, sharpe_ratio

def _select_device() -> torch.device:
    """CUDA > MPS > CPU; override with env DEEPM_DEVICE=cpu|mps|cuda:0."""
    override = os.environ.get("DEEPM_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def empty_device_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


device = _select_device()


class TrainDeepMomentumNetwork:
    """Training harness for Deep Momentum Networks.

    Handles the full training lifecycle: model construction, training loop
    with gradient accumulation, validation with early stopping, test evaluation,
    and checkpoint management. Supports multiple architectures (DeePM, LSTM,
    MomentumTransformer) selected via the ``architecture`` parameter.

    Training uses a differentiable Sharpe ratio loss with optional soft-min
    pooling across the batch for robustness. Results are logged to wandb.
    """

    logger = get_logger(__name__)

    def __init__(
        self,
        train_data: MomentumDataset,
        valid_data: MomentumDataset,
        test_data: MomentumDataset,
        seq_len: int,
        num_features: int,
        save_path: str,
        train_extra_data: List[MomentumDataset] = None,
        **kwargs,
    ):
        """Initialize training harness with train/valid/test datasets."""

        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.train_extra_data = train_extra_data

        self.seq_len = seq_len
        self.num_features = num_features

        for subdir in ("", "models", "returns", "settings", "data-params"):
            os.makedirs(os.path.join(save_path, subdir), exist_ok=True)
        self._save_path = save_path

    def model_save_path(self, run_name: str) -> str:
        """Model checkpoint save path for a given run."""
        return os.path.join(self._save_path, "models", run_name)

    def settings_path(self, run_name: str) -> str:
        """JSON settings path for a given run."""
        return os.path.join(self._save_path, "settings", run_name + ".json")

    def data_params_path(self, run_name: str) -> str:
        """Pickle data-params path for a given run."""
        return os.path.join(self._save_path, "data-params", run_name + ".pkl")

    ARCHITECTURE_CLASSES = {
        "LSTM": LstmBaseline,
        "LSTM_SIMPLE": LstmSimple,
        "DeePM": SpatioTemporalGraphTransformer,
        "AdvancedTemporalBaseline": AdvancedTemporalBaseline,
        "MOM_TRANS": MomentumTransformer,
    }

    def _load_architecture(
        self,
        architecture: str,
        input_dim: int,
        num_tickers: int,
        optimise_loss_function: int,
        **kwargs,
    ) -> torch.nn.Module:
        """Instantiate the model for the given architecture name."""
        cls = self.ARCHITECTURE_CLASSES.get(architecture)
        if cls is None:
            raise ValueError(f"Architecture not recognised: {architecture}")

        if architecture == "DeePM":
            self._inject_deepm_kwargs(kwargs)

        model = cls(
            input_dim=input_dim,
            num_tickers=num_tickers,
            optimise_loss_function=optimise_loss_function,
            **kwargs,
        )
        return model.to(device)

    def _inject_deepm_kwargs(self, kwargs):
        """Add expected_ticker_order and ticker_close_rank for DeePM if not provided."""
        ref_data = self.valid_data if self.valid_data is not None else self.test_data
        if "expected_ticker_order" not in kwargs:
            kwargs["expected_ticker_order"] = list(ref_data.tickers_dict.index)
        if hasattr(ref_data, "ticker_close_rank"):
            kwargs["ticker_close_rank"] = ref_data.ticker_close_rank

    def run(
        self,
        architecture: str,
        lr: float,
        batch_size: int,
        optimise_loss_function: int,
        max_gradient_norm: float,
        weight_decay: float,
        iterations: int,
        early_stopping: int,
        wandb_run_name: str,
        log_wandb: bool = False,
        **kwargs,
    ):
        """Run full train → evaluate pipeline.

        Returns (test_sharpe, best_valid_sharpe, test_sharpe_net, test_calmar, test_calmar_net).
        """
        assert optimise_loss_function == LossFunction.SHARPE.value

        if batch_size == 0:
            batch_size = len(self.train_data)

        kwargs = kwargs.copy()
        kwargs.setdefault("num_tickers", self.valid_data.num_tickers)

        model = self._load_architecture(
            architecture=architecture,
            input_dim=self.num_features,
            optimise_loss_function=optimise_loss_function,
            **kwargs,
        )
        try:
            best_valid_sharpe = self.fit(
                model, lr, batch_size, max_gradient_norm, weight_decay,
                optimise_loss_function, iterations, early_stopping,
                log_wandb, wandb_run_name, **kwargs,
            )

            _, test_sharpe, test_sharpe_net, test_calmar, test_calmar_net = self.predict(
                model, self.model_save_path(wandb_run_name),
                batch_size, optimise_loss_function, DmnMode.INFERENCE,
            )

            self.logger.info("----Iteration update----")
            self.logger.info("Best Valid Sharpe: %.3f", best_valid_sharpe)
            self.logger.info("Test Sharpe Gross: %.3f", test_sharpe)
            self.logger.info("Test Sharpe Net: %.3f", test_sharpe_net)
            self.logger.info("Test Calmar Gross: %.3f", test_calmar)
            self.logger.info("Test Calmar Net: %.3f", test_calmar_net)

            return (
                float(test_sharpe), float(best_valid_sharpe),
                float(test_sharpe_net), float(test_calmar), float(test_calmar_net),
            )
        finally:
            del model
            empty_device_cache()

    @staticmethod
    def get_q(
        current_epoch, warmup_epochs=10, anneal_epochs=20, q_start=2.0, q_end=4.0
    ):
        if current_epoch < warmup_epochs:
            return q_start, False
        elif current_epoch < warmup_epochs + anneal_epochs:
            t = (current_epoch - warmup_epochs) / float(anneal_epochs)
            return q_start + t * (q_end - q_start), False
        else:
            return q_end, True

    def _build_optimizer(self, model, lr, weight_decay, **kwargs):
        """Build Adam/AdamW optimizer with separate learning rate for alpha params."""
        alpha_lr_mult = kwargs.get("alpha_lr_mult", 50.0)
        alpha_names = kwargs.get(
            "alpha_param_suffixes",
            (".alpha", "alpha_gnn", "alpha_ffn", "alpha_attn", "alpha_xatt", "alpha_graph"),
        )

        alpha_params, other_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(name.endswith(suf) for suf in alpha_names):
                alpha_params.append(p)
            else:
                other_params.append(p)

        use_adamw = kwargs.get("use_adamw", False)
        param_groups = [
            {"params": other_params, "lr": lr, "weight_decay": weight_decay if use_adamw else 0.0},
        ]
        if alpha_params:
            param_groups.append(
                {"params": alpha_params, "lr": lr * alpha_lr_mult, "weight_decay": 0.0}
            )

        optimizer = torch.optim.AdamW(param_groups) if use_adamw else torch.optim.Adam(param_groups)

        self.logger.info("alpha params: %d, other params: %d", len(alpha_params), len(other_params))
        if alpha_params:
            self.logger.info("alpha group lr: %s", optimizer.param_groups[-1]["lr"])

        return optimizer

    def _train_epoch(self, model, optimizer, batch_size, max_gradient_norm, q, **kwargs):
        """Run one training epoch and return mean Sharpe."""
        use_contexts = kwargs["use_contexts"]
        cross_section = kwargs["cross_section"]

        if use_contexts and not cross_section:
            self.train_data.shuffle_context()
            self.valid_data.shuffle_context()

        model.train()
        train_dataset = (
            self.train_data
            if not self.train_extra_data
            else torch.utils.data.ConcatDataset([self.train_data, *self.train_extra_data])
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        )

        grad_accum_ratio = int(kwargs.get("grad_accum_ratio", 1))
        if grad_accum_ratio == 1 and kwargs.get("grad_accum_hidden_threshold") is not None:
            grad_accum_ratio = max(1, kwargs["hidden_dim"] // kwargs["grad_accum_hidden_threshold"])

        train_sharpes = []
        for _, samples in enumerate(train_loader):
            if grad_accum_ratio <= 1:
                train_loss = model(
                    **unpack_torch_dataset(
                        samples, self.train_data, device, use_dates_mask=False, live_mode=False
                    ),
                    mode=DmnMode.TRAINING,
                    q=q,
                )
                optimizer.zero_grad()
                train_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
                optimizer.step()
                train_sharpes.append(train_loss.negative().detach().item())
            else:
                train_metric = train_step_exact_chunked(
                    model=model,
                    optimizer=optimizer,
                    samples=samples,
                    train_data=self.train_data,
                    device=device,
                    max_gradient_norm=max_gradient_norm,
                    q=q,
                    grad_accum_ratio=grad_accum_ratio,
                )
                train_sharpes.append(train_metric)

        return np.mean(train_sharpes)

    def fit(
        self,
        model,
        lr,
        batch_size,
        max_gradient_norm,
        weight_decay,
        optimise_loss_function,
        iterations,
        early_stopping,
        log_wandb: bool,
        wandb_run_name: str,
        best_valid_sharpe=-np.inf,
        **kwargs,
    ):
        valid_force_sharpe_loss = kwargs.get("valid_force_sharpe_loss", True)
        if not valid_force_sharpe_loss:
            raise NotImplementedError("valid_force_sharpe_loss=False not supported")

        optimizer = self._build_optimizer(model, lr, weight_decay, **kwargs)

        smooth_alpha = kwargs.get("smooth_alpha", None)
        min_delta = kwargs.get("min_delta", 0.0)
        val_burnin_steps = kwargs.get("val_burnin_steps", 5)
        use_ema = smooth_alpha is not None and 0.0 < smooth_alpha < 1.0

        use_q4_anneal = kwargs.get("q4_anneal", False)
        warmup_epochs = kwargs.get("q4_warmup_epochs", 10)
        anneal_epochs = kwargs.get("q4_anneal_epochs", 20)

        ema_valid = None
        prev_q = None
        burnin_counter = 0
        early_stopping_counter = 0

        for iteration in range(iterations):
            self.logger.info("Iteration %d", iteration + 1)

            # q schedule / annealing
            if use_q4_anneal:
                q, record = self.get_q(
                    iteration, warmup_epochs=warmup_epochs, anneal_epochs=anneal_epochs
                )
            else:
                q = float(kwargs.get("q_fixed", 2.0))
                record = True

            train_loss_metric = "Sharpe" if q == 2.0 else f"Sharpe (q={q:.2f})"

            train_sharpe = self._train_epoch(
                model, optimizer, batch_size, max_gradient_norm, q, **kwargs
            )
            self.logger.info("Train %s: %.3f", train_loss_metric, train_sharpe)

            iteration_valid_sharpe = self._run_validation(
                model, batch_size, valid_force_sharpe_loss, q
            )

            # EMA smoothing of validation metric
            if use_ema:
                if use_q4_anneal and q >= 4.0 and (prev_q is None or prev_q < 4.0):
                    ema_valid = iteration_valid_sharpe
                    burnin_counter = 0
                    self.logger.info("Reset EMA valid metric at first q=%.1f epoch.", q)
                elif ema_valid is None:
                    ema_valid = iteration_valid_sharpe
                else:
                    ema_valid = (
                        smooth_alpha * iteration_valid_sharpe
                        + (1.0 - smooth_alpha) * ema_valid
                    )
                metric_for_early = ema_valid
            else:
                metric_for_early = iteration_valid_sharpe

            prev_q = q

            # Logging
            if log_wandb:
                log_dict = {"train_loss": float(train_sharpe)}
                if use_ema:
                    log_dict["valid_loss"] = float(metric_for_early)
                    log_dict["valid_loss_raw"] = float(iteration_valid_sharpe)
                else:
                    log_dict["valid_loss"] = float(iteration_valid_sharpe)
                wandb.log(log_dict)

            if use_ema:
                self.logger.info(
                    "Valid %s (raw): %.3f | EMA (EW valid): %.3f",
                    train_loss_metric, iteration_valid_sharpe, metric_for_early,
                )
            else:
                self.logger.info("Valid %s: %.3f", train_loss_metric, metric_for_early)

            # Early stopping & best-model selection
            if record:
                ready_for_selection = True
                if burnin_counter < val_burnin_steps:
                    burnin_counter += 1
                    ready_for_selection = False
                    self.logger.info(
                        "Burn-in %d/%d – not updating best model yet.",
                        burnin_counter, val_burnin_steps,
                    )

                if ready_for_selection:
                    if metric_for_early >= best_valid_sharpe + min_delta:
                        torch.save(model.state_dict(), self.model_save_path(wandb_run_name))
                        best_valid_sharpe = metric_for_early
                        early_stopping_counter = 0
                        self.logger.info(
                            "New best valid %s (EW metric): %.3f",
                            train_loss_metric, best_valid_sharpe,
                        )
                    else:
                        early_stopping_counter += 1
                        self.logger.info(
                            "No improvement ≥ %.3f, patience %d/%d",
                            min_delta, early_stopping_counter, early_stopping,
                        )
                        if early_stopping_counter == early_stopping:
                            break

        return best_valid_sharpe

    def _run_validation(self, model, batch_size, force_sharpe_loss, q):
        """Run one full validation pass and return mean Sharpe."""
        valid_sharpes = []
        model.eval()
        valid_loader = torch.utils.data.DataLoader(
            self.valid_data, batch_size=batch_size, drop_last=False
        )
        with torch.no_grad():
            for _, samples in enumerate(valid_loader):
                loss_valid = model(
                    **unpack_torch_dataset(
                        samples,
                        self.valid_data,
                        device,
                        use_dates_mask=False,
                        live_mode=False,
                    ),
                    mode=DmnMode.TRAINING,
                    force_sharpe_loss=force_sharpe_loss,
                    q=q,
                )
                valid_sharpes.append(loss_valid.negative().detach().item())
        return np.mean(valid_sharpes)

    def predict(
        self,
        model: DeepMomentumNetwork,
        model_save_path: str,
        batch_size: int,
        optimise_loss_function: int,
        mode: DmnMode,
    ):
        """Run inference/live predictions and return positions DataFrame."""
        assert mode in [DmnMode.INFERENCE, DmnMode.LIVE]

        # map_location: a checkpoint trained on CUDA must also load on a CPU / MPS machine
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.eval()

        buffers = self._run_inference_loop(model, batch_size, mode)

        if mode is DmnMode.LIVE:
            return self._assemble_live_predictions(buffers)

        predictions = self._assemble_inference_predictions(buffers)
        predictions = self._add_transaction_costs(predictions)
        return self._compute_portfolio_metrics(predictions)

    def _run_inference_loop(self, model, batch_size, mode):
        """Run batched inference and collect raw tensors/arrays."""
        buf = {
            "positions": [], "mask": [], "dates": [], "tickers": [],
            "captured_returns": [], "targets": [],
            "vol_scaling": [], "vol_scaling_prev": [],
        }

        with torch.no_grad():
            for _, samples in enumerate(
                torch.utils.data.DataLoader(
                    self.test_data, batch_size=batch_size, drop_last=False
                )
            ):
                torch_dataset = unpack_torch_dataset(
                    samples, self.test_data, device,
                    use_dates_mask=True, live_mode=False,
                )
                results = model(**torch_dataset, mode=mode)

                if mode is DmnMode.LIVE:
                    positions = results
                    buf["positions"].append(positions.detach().cpu())
                else:
                    captured_return, positions = results
                    buf["captured_returns"].append(captured_return.detach().cpu())
                    buf["positions"].append(positions.detach().cpu())

                date_mask = torch_dataset["date_mask"]
                target_tickers = torch_dataset["target_tickers"]
                dates = torch_dataset["dates"]

                if model.is_cross_section:
                    date_mask = model.combine_batch_and_asset_dim(date_mask, 3)
                    target_tickers = model.combine_batch_and_asset_dim(target_tickers, 2)
                    dates = model.combine_batch_and_asset_dim(dates, 3)

                if mode is not DmnMode.LIVE:
                    target_y = torch_dataset["target_y"]
                    vol_scaling = torch_dataset["vol_scaling_amount"]
                    vol_scaling_prev = torch_dataset["vol_scaling_amount_prev"]

                    if model.is_cross_section:
                        target_y = model.combine_batch_and_asset_dim(target_y, 4)
                        vol_scaling = model.combine_batch_and_asset_dim(vol_scaling, 3)
                        vol_scaling_prev = model.combine_batch_and_asset_dim(vol_scaling_prev, 3)

                    n = positions.shape[1]
                    buf["targets"].append(target_y[:, -n:, -1].detach().cpu())
                    buf["vol_scaling"].append(vol_scaling[:, -n:].detach().cpu())
                    buf["vol_scaling_prev"].append(vol_scaling_prev[:, -n:].detach().cpu())

                n = positions.shape[1]
                buf["mask"].append(date_mask[:, -n:].detach().cpu())
                buf["tickers"].append(target_tickers.detach().cpu())

                if model.is_cross_section:
                    buf["dates"].append(
                        self.test_data.dates_string[
                            dates[:, -n:].detach().cpu().numpy().astype(int)
                        ]
                    )
                else:
                    buf["dates"].append(np.array([*dates]).T[:, -n:])

        return buf

    def _flatten_common_buffers(self, buf):
        """Flatten mask, dates, tickers, and positions from inference buffers."""
        mask = torch.cat(buf["mask"]).flatten().numpy()
        dates = np.concatenate(buf["dates"]).flatten()
        raw_tickers = torch.cat(buf["tickers"]).flatten().numpy()

        tick_mapping = self.test_data.tickers_from_idx_dict
        n_steps = buf["positions"][-1].shape[1]
        tickers = sum([[tick_mapping[t]] * n_steps for t in raw_tickers], [])
        positions = torch.cat(buf["positions"]).flatten().numpy()

        return mask, dates, tickers, positions

    def _assemble_live_predictions(self, buf):
        """Build live-mode predictions DataFrame."""
        mask, dates, tickers, positions = self._flatten_common_buffers(buf)
        return pd.DataFrame(
            {"ticker": tickers, "position": positions}, index=dates,
        )[mask]

    def _assemble_inference_predictions(self, buf):
        """Build inference-mode predictions DataFrame with cost columns."""
        mask, dates, tickers, positions = self._flatten_common_buffers(buf)
        return pd.DataFrame(
            {
                "ticker": tickers,
                "captured_return": torch.cat(buf["captured_returns"]).flatten().numpy(),
                "position": positions,
                "target_return": torch.cat(buf["targets"]).flatten().numpy(),
                "cost_scaling": torch.cat(buf["vol_scaling"]).flatten().numpy(),
                "cost_scaling_prev": torch.cat(buf["vol_scaling_prev"]).flatten().numpy(),
            },
            index=dates,
        )[mask]

    def _add_transaction_costs(self, predictions_by_asset: pd.DataFrame) -> pd.DataFrame:
        """Merge transaction costs and compute net captured returns."""
        predictions_by_asset = (
            predictions_by_asset.reset_index()
            .merge(
                self.test_data.ticker_ref[["ticker", "transaction_cost"]],
                on="ticker",
            )
            .rename(columns={"index": ""})
            .set_index("")
        )

        predictions_by_asset["cost_scaling"] *= (
            predictions_by_asset["transaction_cost"].values * 1e-4
        )
        predictions_by_asset["cost_scaling_prev"] *= (
            predictions_by_asset["transaction_cost"].values * 1e-4
        )

        predictions_by_asset["pos_prev"] = (
            predictions_by_asset.groupby("ticker")["position"].shift(1).fillna(0.0)
        )
        predictions_by_asset["cost"] = (
            predictions_by_asset["cost_scaling"] * predictions_by_asset["position"]
            - predictions_by_asset["cost_scaling_prev"]
            * predictions_by_asset["pos_prev"]
        ).abs()

        predictions_by_asset["captured_return_net"] = (
            predictions_by_asset["captured_return"] - predictions_by_asset["cost"]
        )
        return predictions_by_asset

    @staticmethod
    def _compute_portfolio_metrics(predictions_by_asset: pd.DataFrame):
        """Aggregate per-asset returns to portfolio level and compute Sharpe/Calmar."""
        portfolio = predictions_by_asset.copy()
        portfolio.index.name = "idx"
        portfolio = portfolio.groupby("idx")[
            ["captured_return", "captured_return_net"]
        ].sum()

        gross_series = portfolio["captured_return"]
        net_series = portfolio["captured_return_net"]

        return (
            predictions_by_asset,
            sharpe_ratio(gross_series),
            sharpe_ratio(net_series),
            calmar_ratio(gross_series / np.sqrt(252)),
            calmar_ratio(net_series / np.sqrt(252)),
        )

