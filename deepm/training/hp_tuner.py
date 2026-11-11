import copy
import gc
import glob
import json
import os
import pickle
import re

import numpy as np
import pandas as pd
import torch

import wandb

from deepm.training.train import TrainDeepMomentumNetwork, empty_device_cache
from deepm.utils.logging_utils import get_logger
from deepm.configs.settings import WANDB_ENTITY


CONFIGS_IGNORE_WANDB = [
    "project",
    "data_yaml",
    "ticker_reference_file",
    "test_run",
    "feature_prepare",
    "versions",
    "first_train_year",
    "final_test_year",
    "first_test_year",
    "reference_cols",
    "target",
    "targets",
    "target_vol",
    "description",
    "run_name",
    "extra_data_pre_steps",
    "vs_factor_scaler",
    "trans_cost_scaler",
]

SUMMARY_KEYS = [
    "valid_loss_best",
    "test_sharpe_gross",
    "test_sharpe_net",
    "test_calmar_gross",
    "test_calmar_net",
]


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class Tuner:
    """Hyperparameter tuner that runs random-search sweeps with wandb."""

    logger = get_logger(__name__)

    def __init__(
        self,
        settings,
        sweep_settings,
        architecture,
        train_data,
        valid_data,
        test_data,
        test_start,
        test_end,
        train_extra_data,
        data_params,
        valid_end_optional: int = None,
    ) -> None:
        self.settings = settings
        self.architecture = architecture
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.test_start = test_start
        self.test_end = test_end
        self.data_params = data_params
        self.valid_end_optional = valid_end_optional
        self.iteration = 0

        self.sweep_settings = {
            "name": f"{settings['description']}_{architecture}_{test_start}",
            **copy.deepcopy(sweep_settings),
        }
        self.save_path = os.path.join(
            settings["save_directory"],
            settings["description"],
            settings["run_name"],
            f"{test_start}",
        )
        self.model_training = TrainDeepMomentumNetwork(
            train_data, valid_data, test_data,
            settings["seq_len"],
            len(settings["features"]),
            save_path=self.save_path,
            train_extra_data=train_extra_data,
        )

    # ── Sweep orchestration ───────────────────────────────────────

    def hyperparameter_optimisation(self):
        self.iteration = 0
        if self._use_local_sweep():
            return self._local_hyperparameter_optimisation()
        self._ensure_wandb_login()

        sweep_id, num_completed = self._resume_or_create_sweep()
        remaining = self.settings["random_search_max_iterations"] - num_completed
        if remaining <= 0:
            return self._save_sweep_results(sweep_id)

        self._fixed_config = self._build_fixed_config()

        wandb.agent(
            f"{WANDB_ENTITY}/{self.settings['project']}/{sweep_id}",
            function=self._run_single_trial,
            count=remaining,
        )
        return self._save_sweep_results(sweep_id)

    # ── Local random search (no wandb server needed) ──────────────

    def _use_local_sweep(self) -> bool:
        """Sweeps need the wandb server; fall back to a local random search
        when wandb is offline/disabled or ``local_sweep: true`` is set."""
        return bool(self.settings.get("local_sweep")) or os.environ.get(
            "WANDB_MODE", ""
        ).lower() in ("offline", "disabled", "dryrun")

    def _sample_hp(self, rng: np.random.Generator) -> dict:
        hp = {}
        for k, spec in self.sweep_settings["parameters"].items():
            if "values" in spec:
                vals = spec["values"]
                hp[k] = vals[int(rng.integers(len(vals)))]
            elif "value" in spec:
                hp[k] = spec["value"]
            else:
                raise NotImplementedError(f"Local sweep only supports value/values: {k}={spec}")
            if isinstance(hp[k], np.generic):
                hp[k] = hp[k].item()
        if "max_gradient_norm" in hp:
            hp["max_gradient_norm"] = float(hp["max_gradient_norm"])
        return hp

    def _local_hyperparameter_optimisation(self):
        """Random search equivalent to the wandb sweep. Trial i uses seed ``local_sweep_seed + i`` for both
        HP sampling and torch init. Each finished trial is written atomically to ``local_trials/local-<i>.json``
        (status ok/failed). A trial counts as done only if its
        summary metrics are finite and its seed matches, so an interrupted search resumes, a trial that raises
        is recorded and skipped (``DEEPM_RETRY_FAILED=1`` retries it) instead of stopping the search, and rows
        from another sweep/seed or with i >= random_search_max_iterations are ignored.

        Parallel shards: with env ``DEEPM_SHARD=k/n`` this process runs only the trials with ``i % n == k``,
        so n processes can share one GPU (``scripts/run_sharded.sh``). Shards never write
        all_runs/best_runs; one final run without DEEPM_SHARD aggregates (it finds every trial done)."""
        os.environ.setdefault("WANDB_MODE", "disabled")
        shard = os.environ.get("DEEPM_SHARD")
        k, n = (int(x) for x in shard.split("/")) if shard else (0, 1)
        if not 0 <= k < n:
            raise ValueError(f"DEEPM_SHARD must be k/n with 0 <= k < n, got {shard!r}")
        total = self.settings["random_search_max_iterations"]
        base_seed = int(self.settings.get("local_sweep_seed", 0))
        done = self._local_trial_records(base_seed)
        failed = set() if os.environ.get("DEEPM_RETRY_FAILED") == "1" else self._failed_local_trials()
        todo = [i for i in range(total)
                if i % n == k and f"local-{i}" not in done and f"local-{i}" not in failed]
        self.logger.info(
            "Local random search: %s/%s trials done (%s failed), this process (shard %s/%s) runs %s (%s)",
            sum(int(nm.split("-")[1]) < total for nm in done), total, len(failed), k, n, len(todo), self.save_path,
        )

        for i in todo:
            name = f"local-{i}"
            seed = base_seed + i
            hp = self._sample_hp(np.random.default_rng(seed))
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.iteration = i + 1
            self.logger.info("----Local random search iteration %s/%s (%s)----", i + 1, total, name)
            self.logger.info("----%s---- hp=%s", self.settings["description"], hp)
            try:
                (
                    test_sharpe, valid_sharpe, test_sharpe_net,
                    test_calmar, test_calmar_net,
                ) = self.model_training.run(
                    architecture=self.architecture,
                    log_wandb=False,
                    wandb_run_name=name,
                    **self.settings,
                    **hp,
                )
                self._save_trial_artifacts(name, hp, test_sharpe, test_sharpe_net, valid_sharpe)
                rec = {
                    "status": "ok",
                    "valid_loss_best": valid_sharpe,
                    "test_sharpe_gross": test_sharpe,
                    "test_sharpe_net": test_sharpe_net,
                    "test_calmar_gross": test_calmar,
                    "test_calmar_net": test_calmar_net,
                    "seed": seed,
                    **hp,
                }
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:  # record and move on: one bad draw (e.g. OOM) must not block the search
                self.logger.exception("trial %s failed", name)
                rec = {"status": "failed", "error": repr(e)[:500], "seed": seed, **hp}
            finally:
                gc.collect()
                empty_device_cache()
            self._write_local_trial(name, rec)

        if shard:  # shards would race on all_runs/best_runs; the final unsharded run aggregates
            return None
        return self._save_local_results(total, base_seed)

    def _write_local_trial(self, name, rec):
        """Atomic write (tmp file + os.replace): a killed process leaves no half-written record."""
        d = os.path.join(self.save_path, "local_trials")
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, f".{name}.json.tmp{os.getpid()}")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f, cls=NumpyEncoder)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, os.path.join(d, f"{name}.json"))

    def _read_local_trials(self):
        """name -> record from local_trials/local-*.json."""
        recs = {}
        for path in sorted(glob.glob(os.path.join(self.save_path, "local_trials", "local-*.json"))):
            try:
                with open(path, encoding="utf-8") as f:
                    recs[os.path.basename(path)[:-5]] = json.load(f)
            except (OSError, ValueError) as e:
                self.logger.warning("skipping unreadable trial record %s: %s", path, e)
        return recs

    @staticmethod
    def _valid_local_trial(name, rec, base_seed):
        m = re.fullmatch(r"local-(\d+)", name)
        if not m or rec.get("status", "ok") != "ok":
            return False
        try:
            vals = [float(rec[key]) for key in SUMMARY_KEYS]
            seed_ok = int(float(rec["seed"])) == base_seed + int(m.group(1))
        except (KeyError, TypeError, ValueError):
            return False
        return seed_ok and bool(np.all(np.isfinite(vals)))

    def _local_trial_records(self, base_seed):
        return {nm: r for nm, r in self._read_local_trials().items() if self._valid_local_trial(nm, r, base_seed)}

    def _failed_local_trials(self):
        return {nm for nm, r in self._read_local_trials().items() if r.get("status") == "failed"}

    def _save_local_results(self, total, base_seed):
        recs = {nm: r for nm, r in self._local_trial_records(base_seed).items() if int(nm.split("-")[1]) < total}
        if not recs:
            raise RuntimeError(f"no completed trials in {self.save_path}")
        if len(recs) < total:
            self.logger.warning("only %s of %s trials completed in %s", len(recs), total, self.save_path)
        all_runs = pd.DataFrame.from_dict(
            {nm: {key: float(r[key]) for key in SUMMARY_KEYS} for nm, r in recs.items()}, orient="index"
        ).sort_values("valid_loss_best", ascending=False)
        best_runs = all_runs.head(self.settings["top_n_seeds"])
        all_runs.to_csv(os.path.join(self.save_path, "all_runs.csv"))
        best_runs.to_csv(os.path.join(self.save_path, "best_runs.csv"))
        best_means = best_runs.mean()
        best_means.to_json(os.path.join(self.save_path, "best_runs_mean.json"), indent=4)
        return (
            best_means.loc["test_sharpe_gross"],
            best_means.loc["test_sharpe_net"],
            best_means.loc["valid_loss_best"],
            best_means.loc["test_calmar_gross"],
            best_means.loc["test_calmar_net"],
        )

    @staticmethod
    def _ensure_wandb_login():
        if wandb.api.api_key:
            return
        key_kwargs = (
            {"key": os.environ["WANDB_API_KEY"]}
            if "WANDB_API_KEY" in os.environ
            else {}
        )
        wandb.login(**key_kwargs)

    def _resume_or_create_sweep(self):
        """Find an existing sweep or create a new one. Returns (sweep_id, num_completed)."""
        api = wandb.Api()
        runs = api.runs(
            f"{WANDB_ENTITY}/{self.settings['project']}",
            filters={"group": self.sweep_settings["name"]},
            per_page=10000,
        )
        if len(runs) == 0:
            sweep_id = wandb.sweep(
                sweep=self.sweep_settings,
                project=self.settings["project"],
                entity=WANDB_ENTITY,
            )
            return sweep_id, 0

        sweep_id = runs[0].sweep.id
        assert all(r.sweep.id == sweep_id for r in runs)
        num_completed = sum(1 for r in runs if r.state == "finished")
        return sweep_id, num_completed

    def _build_fixed_config(self):
        return {
            "model": self.settings["description"],
            "architecture": self.architecture,
            "test_start": self.test_start,
            "train_start": self.settings["first_train_year"],
            "valid_end_optional": self.valid_end_optional,
            "train_tickers": self.valid_data.tickers_dict.index.tolist(),
            **{k: v for k, v in self.settings.items() if k not in CONFIGS_IGNORE_WANDB},
        }

    # ── Single trial ──────────────────────────────────────────────

    def _run_single_trial(self, config: dict = None):
        """Execute one wandb sweep trial: train, evaluate, and save artifacts."""
        try:
            with wandb.init(group=self.sweep_settings["name"], config=config):
                config = wandb.config
                name = wandb.run.name

                hp = dict(config.items())
                if "max_gradient_norm" in hp:
                    hp["max_gradient_norm"] = float(hp["max_gradient_norm"])

                for k, v in self._fixed_config.items():
                    config[k] = v

                self.iteration += 1
                self.logger.info("----Random grid search iteration %s----", self.iteration)
                self.logger.info("----%s----", self.settings["description"])

                (
                    test_sharpe, valid_sharpe, test_sharpe_net,
                    test_calmar, test_calmar_net,
                ) = self.model_training.run(
                    architecture=self.architecture,
                    log_wandb=True,
                    wandb_run_name=name,
                    **self.settings,
                    **hp,
                )

                self._save_trial_artifacts(name, hp, test_sharpe, test_sharpe_net, valid_sharpe)

                wandb.log({
                    "test_sharpe_gross": test_sharpe,
                    "test_sharpe_net": test_sharpe_net,
                    "valid_loss_best": valid_sharpe,
                    "test_calmar_gross": test_calmar,
                    "test_calmar_net": test_calmar_net,
                }, step=None)
        finally:
            gc.collect()
            empty_device_cache()

    def _save_trial_artifacts(self, name, hp, test_sharpe, test_sharpe_net, valid_sharpe):
        """Save settings JSON and data params pickle for a completed trial."""
        with open(self.model_training.settings_path(name), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "test_sharpe": test_sharpe,
                    "test_sharpe_net": test_sharpe_net,
                    "valid_sharpe": valid_sharpe,
                    **hp,
                    **self.settings,
                    "tickers_dict": self.valid_data.tickers_dict.to_dict(),
                    "num_tickers": self.train_data.num_tickers,
                    "num_tickers_full_univ": self.valid_data.num_tickers_full_univ,
                },
                f, indent=4, cls=NumpyEncoder,
            )

        with open(self.model_training.data_params_path(name), "wb") as f:
            pickle.dump(self.data_params, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ── Results aggregation ───────────────────────────────────────

    def _save_sweep_results(self, sweep_id):
        """Aggregate finished runs from a wandb sweep, save CSVs, and return best metrics."""
        sweep = wandb.Api().sweep(
            f"{WANDB_ENTITY}/{self.settings['project']}/{sweep_id}"
        )
        runs = [r for r in sweep.runs if r.state == "finished"]

        all_runs = pd.concat(
            [
                pd.Series(r.summary._json_dict, name=r.name).loc[SUMMARY_KEYS]
                for r in runs
                if all(k in r.summary._json_dict for k in SUMMARY_KEYS)
            ],
            axis=1,
        ).T.sort_values("valid_loss_best", ascending=False)

        best_runs = all_runs.head(self.settings["top_n_seeds"])

        all_runs.to_csv(os.path.join(self.save_path, "all_runs.csv"))
        best_runs.to_csv(os.path.join(self.save_path, "best_runs.csv"))

        best_means = best_runs.mean()
        best_means.to_json(
            os.path.join(self.save_path, "best_runs_mean.json"), indent=4,
        )

        wandb.finish()
        return (
            best_means.loc["test_sharpe_gross"],
            best_means.loc["test_sharpe_net"],
            best_means.loc["valid_loss_best"],
            best_means.loc["test_calmar_gross"],
            best_means.loc["test_calmar_net"],
        )
