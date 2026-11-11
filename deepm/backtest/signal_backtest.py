"""Job for signal backtesting."""

import importlib
import json
import os
from copy import deepcopy
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deepm._paths import DATA_DIR, DIAGNOSTICS_DIR
from deepm.backtest.metrics import (
    TARGET_VOL,
    _annualized_sharpe,
    calculate_metrics_and_rescale,
)
from deepm.data.dataset import get_transaction_costs
from deepm.utils.logging_utils import get_logger


class SignalBacktestJob:
    """Generate trading signals for a list of tickers and optionally run backtest diagnostics."""

    logger = get_logger(__name__)

    def __init__(
        self,
        name: str,
        tickers: List[str],
        start_date: str,
        data_parquet: str,
        save_dir: str,
        cfg_model: dict,
        cfg_train: dict,
        transaction_cost_path: str,
        variable_importance_mode: bool = False,
        output_signal_weights: bool = False,
        directory_already_altered: bool = False,
        end_date: str = None,
        ticker_mapping: dict = None,
    ):
        self.name = name
        self.cfg_model = deepcopy(cfg_model)
        self.cfg_train = deepcopy(cfg_train)
        self.ticker_mapping = ticker_mapping or {}
        self.tickers = tickers
        if self.cfg_train.get("ticker_subset", None) is not None:
            self.tickers = list(
                set(self.tickers).intersection(set(self.cfg_train["ticker_subset"]))
            )
        if ticker_mapping:
            self.tickers = list(map(ticker_mapping.get, self.tickers, self.tickers))

        self.start_date = start_date
        self.end_date = end_date
        self.data_parquet = data_parquet
        self.save_dir = save_dir

        self.transaction_cost_path = transaction_cost_path
        self.variable_importance_mode = variable_importance_mode
        if self.variable_importance_mode and not directory_already_altered:
            self.save_dir = os.path.join(self.save_dir, "variable_importance")
        self.output_signal_weights = output_signal_weights

        if self.output_signal_weights and not directory_already_altered:
            self.save_dir = os.path.join(self.save_dir, "signal_weights")

        self.load_data_path = str(DATA_DIR / ("feats-" + data_parquet))

    def _instantiate_model(self):
        """Build a model instance from the dynamic module/class config."""
        cfg = deepcopy(self.cfg_model)
        cfg["prediction_folder"] = self.save_dir
        module_name = cfg.pop("module")
        class_name = cfg.pop("class")
        smooth_signal_span = cfg.pop("smooth_signal_span", None)
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls(**cfg), smooth_signal_span

    def generate_signals(
        self,
        save: bool = True,
    ):
        os.makedirs(self.save_dir, exist_ok=True)
        model, smooth_signal_span = self._instantiate_model()

        self.dataframe = pd.read_parquet(self.load_data_path)
        if self.ticker_mapping:  # mapping with an empty dict would turn every ticker into NaN
            self.dataframe["ticker"] = self.dataframe["ticker"].map(self.ticker_mapping)

        extra_params = {}
        if self.variable_importance_mode:
            extra_params["variable_importance_mode"] = True
        if self.output_signal_weights:
            extra_params["output_signal_weights"] = True

        predictions = model.predict(
            self.dataframe, self.tickers, self.start_date, **extra_params
        )
        predictions.index = pd.to_datetime(predictions.index)
        if self.end_date is not None:
            predictions = predictions.loc[: self.end_date]

        if self.output_signal_weights:
            if predictions.index.name != "date":
                predictions.index.name = "date"
            predictions = predictions.reset_index().set_index(["date", "ticker"])

        if smooth_signal_span:
            predictions = predictions.ewm(span=smooth_signal_span).mean()

        if save:
            self.save_predictions(
                predictions, self.save_dir,
                self.variable_importance_mode or self.output_signal_weights,
            )

        return predictions

    @classmethod
    def save_predictions(
        cls,
        dataframe: pd.DataFrame,
        save_dir: str,
        stacked_mode: bool = False,
    ):
        os.makedirs(save_dir, exist_ok=True)
        if stacked_mode:
            for ticker, df_slice in dataframe.groupby("ticker"):
                df_slice.to_parquet(os.path.join(save_dir, f"{ticker}.parquet"))
        else:
            for ticker in dataframe.columns:
                dataframe[[ticker]].to_parquet(os.path.join(save_dir, f"{ticker}.parquet"))

    @classmethod
    def calc_returns(
        cls,
        dataframe: pd.DataFrame,
        transaction_costs: pd.DataFrame,
        price_series: pd.DataFrame,
        scale_to_target_vol: bool = True,
    ) -> pd.DataFrame:
        reference = (
            price_series.reset_index()
            .rename(columns={"index": "date"})
            .merge(transaction_costs, on="ticker")
        )
        reference["date"] = pd.to_datetime(reference["date"].dt.date)
        positions = (
            dataframe.stack()
            .reset_index()
            .rename(columns={"level_0": "date", "level_1": "ticker", 0: "position"})
        )
        positions["date"] = pd.to_datetime(positions["date"])
        positions = positions.merge(reference, on=["ticker", "date"])

        # Calculate Scaled Turnover for Cost
        positions["holdings_x_transaction"] = (
            positions["position"]
            * positions["vs_factor"]
            * positions["transaction_cost"]
            * 1e-4
        )

        # Turnover Cost
        positions["cost"] = (
            positions.groupby("ticker")["holdings_x_transaction"]
            .diff()
            .fillna(0.0)
            .abs()
        )

        # Gross and Net Returns
        positions["gross_return"] = positions["position"] * positions["target"]
        positions["net_return"] = positions["gross_return"] - positions["cost"]

        if scale_to_target_vol:
            # Scales ticker-level returns; portfolio-level scaling is applied later
            positions[
                [
                    "target",
                    "vs_factor",
                    "holdings_x_transaction",
                    "cost",
                    "gross_return",
                    "net_return",
                ]
            ] *= TARGET_VOL / np.sqrt(252)

        return positions

    def calc_group_returns(
        self,
        dataframe: pd.DataFrame,
        transaction_costs: pd.DataFrame,
        price_series: pd.DataFrame,
        ticker_groups: dict,
        scale_to_target_vol: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate returns aggregated by macro group.
        """
        # Calculate raw asset returns
        positions = self.calc_returns(
            dataframe, transaction_costs, price_series, scale_to_target_vol=False
        )

        # Map tickers to groups
        positions["group"] = positions["ticker"].map(ticker_groups)

        # Filter out positions that don't belong to any defined group
        positions = positions.dropna(subset=["group"])

        if positions.empty:
            return None

        # Mean return per group per date (equivalent to sum / n_tickers)
        group_returns = positions.groupby(["date", "group"])[
            ["gross_return", "net_return"]
        ].mean()

        if scale_to_target_vol:
            # Scale each group series to target vol
            def rescale(x):
                vol = x.std() * np.sqrt(252)
                if vol == 0 or np.isnan(vol):
                    return x
                return x * (TARGET_VOL / vol)

            # Use transform to keep shape
            group_returns = group_returns.groupby(level="group").transform(rescale)

        return group_returns

    def save_and_plot_diagnostics(
        self,
        portfolio_returns: pd.DataFrame,
        positions_df: pd.DataFrame,
        diagnostics_dir: str,
        group_returns: pd.DataFrame = None,
        reference_data: pd.DataFrame = None,
        transaction_costs: pd.DataFrame = None,
    ):
        """
        Calculates full metrics, saves to JSON/CSV, and generates plots.
        """
        os.makedirs(diagnostics_dir, exist_ok=True)
        plt.ioff()

        ann_turnover_raw, avg_gmv_raw = self._compute_turnover(positions_df)

        # --- 2. Calculate Metrics & Rescale Returns (Strategy) ---
        metrics_gross, rescaled_gross = calculate_metrics_and_rescale(
            portfolio_returns["gross_return"], turnover=None
        )
        metrics_net, rescaled_net = calculate_metrics_and_rescale(
            portfolio_returns["net_return"],
            turnover=ann_turnover_raw,
            avg_gmv=avg_gmv_raw,
        )

        # --- 3. Construct Passive Benchmark (Equal Risk) ---
        if reference_data is not None and transaction_costs is not None:
            passive_df = (
                positions_df[["date", "ticker"]].copy().set_index(["date", "ticker"])
            )
            passive_df["position"] = 1.0
            passive_wide = passive_df["position"].unstack()

            passive_pos = self.calc_returns(
                passive_wide,
                transaction_costs,
                reference_data,
                scale_to_target_vol=False,
            )
            passive_port = (
                passive_pos.groupby("date")[["gross_return", "net_return"]].sum()
                / passive_pos["ticker"].nunique()
            )
            metrics_passive, rescaled_passive = calculate_metrics_and_rescale(
                passive_port["net_return"], turnover=None
            )
        else:
            metrics_passive, rescaled_passive = {}, None

        # --- 4. Compare vs Benchmark (Add to Strategy Metrics) ---
        if rescaled_passive is not None:
            metrics_net_bench, _ = calculate_metrics_and_rescale(
                portfolio_returns["net_return"],
                turnover=ann_turnover_raw,
                avg_gmv=avg_gmv_raw,
                benchmark_returns=rescaled_passive,
            )
            metrics_net.update(metrics_net_bench)

        # --- 4b. Metrics Ex-COVID (2020-2021 excluded) ---
        ex_covid_mask = ~(
            (portfolio_returns.index.year == 2020)
            | (portfolio_returns.index.year == 2021)
        )

        if ex_covid_mask.sum() > 20:  # Ensure enough data remains
            # A. Ex-COVID Gross Sharpe (Requested)
            metrics_gross_excovid, _ = calculate_metrics_and_rescale(
                portfolio_returns.loc[ex_covid_mask, "gross_return"], turnover=None
            )
            metrics_net["Ex-COVID Gross Sharpe Ratio"] = metrics_gross_excovid[
                "Sharpe Ratio"
            ]

            # B. Ex-COVID Net Performance (Including vs Benchmark / Alpha tests)
            bench_excovid = (
                rescaled_passive.loc[ex_covid_mask]
                if rescaled_passive is not None
                else None
            )

            metrics_net_excovid, _ = calculate_metrics_and_rescale(
                portfolio_returns.loc[ex_covid_mask, "net_return"],
                turnover=None,
                benchmark_returns=bench_excovid,
            )

            # Prefix all sub-period metrics for storage
            metrics_net_excovid_prefixed = {
                f"Ex-COVID {k}": v for k, v in metrics_net_excovid.items()
            }
            metrics_net.update(metrics_net_excovid_prefixed)

        # --- 5. Save Metrics to JSON ---
        full_metrics = {
            "gross": metrics_gross,
            "net": metrics_net,
            "passive_benchmark": metrics_passive,
        }
        with open(os.path.join(diagnostics_dir, "metrics.json"), "w") as f:
            json.dump(full_metrics, f, indent=4)

        # --- 6. Save PnL Series ---
        wealth_gross = (1 + rescaled_gross).cumprod()
        wealth_net = (1 + rescaled_net).cumprod()

        pnl_data = {
            "gross_wealth_index": wealth_gross,
            "net_wealth_index": wealth_net,
            "gross_return_rescaled": rescaled_gross,
            "net_return_rescaled": rescaled_net,
        }

        if rescaled_passive is not None:
            wealth_passive = (1 + rescaled_passive).cumprod()
            pnl_data["passive_wealth_index"] = wealth_passive
            pnl_data["passive_return_rescaled"] = rescaled_passive

        pnl_df = pd.DataFrame(pnl_data)
        pnl_df.to_csv(os.path.join(diagnostics_dir, "pnl_curves.csv"))

        self._generate_plots(
            diagnostics_dir, metrics_gross, metrics_net, metrics_passive,
            rescaled_gross, rescaled_net, rescaled_passive, group_returns,
        )

    @staticmethod
    def _compute_turnover(positions_df: pd.DataFrame):
        """Compute annualized turnover and average gross market value."""
        positions_df = positions_df.sort_values(["ticker", "date"])
        positions_df["notional"] = positions_df["position"] * positions_df["vs_factor"]
        positions_df["daily_turnover"] = (
            positions_df.groupby("ticker")["notional"].diff().abs().fillna(0.0)
        )

        num_assets = positions_df["ticker"].nunique()
        avg_daily_turnover = (
            positions_df.groupby("date")["daily_turnover"].sum() / num_assets
        ).mean()
        ann_turnover_raw = avg_daily_turnover * 252.0

        positions_df["abs_notional"] = positions_df["notional"].abs()
        avg_gmv_raw = (
            positions_df.groupby("date")["abs_notional"].sum() / num_assets
        ).mean()

        return ann_turnover_raw, avg_gmv_raw

    def _generate_plots(
        self, diagnostics_dir, metrics_gross, metrics_net, metrics_passive,
        rescaled_gross, rescaled_net, rescaled_passive, group_returns,
    ):
        """Generate and save PnL, annual Sharpe, and group return plots."""
        wealth_gross = (1 + rescaled_gross).cumprod()
        wealth_net = (1 + rescaled_net).cumprod()

        fig, axis = plt.subplots(tight_layout=True, figsize=(10, 6))
        wealth_gross.plot(
            ax=axis,
            label=f"Gross (SR: {metrics_gross['Sharpe Ratio']:.2f})",
            alpha=0.4, linestyle="--",
        )
        wealth_net.plot(
            ax=axis, label=f"Net (SR: {metrics_net['Sharpe Ratio']:.2f})", linewidth=1.5,
        )
        if rescaled_passive is not None:
            wealth_passive = (1 + rescaled_passive).cumprod()
            wealth_passive.plot(
                ax=axis,
                label=f"Passive Equal Risk (SR: {metrics_passive.get('Sharpe Ratio', 0):.2f})",
                alpha=0.6, color="gray",
            )
        axis.set_title(f"{self.name} - Cumulative Wealth (10% Vol Target)")
        axis.set_ylabel("Wealth Index (Start=1.0)")
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.savefig(os.path.join(diagnostics_dir, "pnl_cumulative.png"))
        plt.close(fig)

        fig, axis = plt.subplots(tight_layout=True)
        ann_gross = rescaled_gross.groupby(rescaled_gross.index.year).apply(_annualized_sharpe)
        ann_net = rescaled_net.groupby(rescaled_net.index.year).apply(_annualized_sharpe)
        plot_data = {"Gross": ann_gross, "Net": ann_net}
        if rescaled_passive is not None:
            plot_data["Passive"] = rescaled_passive.groupby(
                rescaled_passive.index.year
            ).apply(_annualized_sharpe)
        pd.DataFrame(plot_data).plot.bar(ax=axis, grid=True)
        axis.set_title("Annualized Sharpe Ratio (Rescaled)")
        fig.savefig(os.path.join(diagnostics_dir, "annual_sharpe.png"))
        plt.close(fig)

        if group_returns is not None:
            gr = group_returns.reset_index()
            gr["date"] = pd.to_datetime(gr["date"])
            for col, title in [("gross_return", "Gross"), ("net_return", "Net")]:
                fig, axis = plt.subplots(tight_layout=True, figsize=(10, 6))
                pivoted = gr.pivot(
                    index="date", columns="group", values=col,
                ).fillna(0)
                wealth = (1 + pivoted).cumprod()
                for group_name in wealth.columns:
                    wealth[group_name].plot(ax=axis, label=group_name, linewidth=1.5)
                axis.set_title(f"{self.name} - {title} Cumulative Wealth by Group")
                axis.set_ylabel("Wealth Index (Start=1.0)")
                axis.grid(True, alpha=0.3)
                axis.legend()
                fig.savefig(os.path.join(diagnostics_dir, f"group_{col.split('_')[0]}_pnl.png"))
                plt.close(fig)

    def _load_reference_data(self):
        """Load reference price/vol data for diagnostics."""
        return self.dataframe[["ticker", "target", "vs_factor"]]

    def _run_diagnostics(self, predictions, ticker_groups):
        """Compute metrics, generate plots, and save diagnostic outputs."""
        diagnostics_dir = str(DIAGNOSTICS_DIR / self.name)

        reference = self._load_reference_data()
        transaction_costs = get_transaction_costs(self.transaction_cost_path)
        if self.ticker_mapping:
            transaction_costs["ticker"] = transaction_costs["ticker"].map(
                self.ticker_mapping
            )

        positions = self.calc_returns(
            predictions, transaction_costs, reference, scale_to_target_vol=False
        )
        portfolio_returns = (
            positions.groupby("date")[["gross_return", "net_return"]].sum()
            / positions["ticker"].nunique()
        )
        group_returns = self.calc_group_returns(
            predictions, transaction_costs, reference,
            ticker_groups=ticker_groups, scale_to_target_vol=True,
        )

        self.save_and_plot_diagnostics(
            portfolio_returns, positions, diagnostics_dir,
            group_returns, reference_data=reference, transaction_costs=transaction_costs,
        )

    def main(
        self,
        diagnostics: bool = False,
        ticker_groups: dict = None,
    ):
        predictions = self.generate_signals(save=True)

        if diagnostics:
            self._run_diagnostics(predictions, ticker_groups or {})


