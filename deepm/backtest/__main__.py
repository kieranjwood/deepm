"""Unified CLI for single-model and ensemble backtesting.

Usage::

    python -m deepm.backtest --name bt-deepm-gat --diagnostics
    python -m deepm.backtest --name my-ensemble --ensemble --diagnostics
"""

import argparse
import os
import sys
import time

import yaml

from deepm._paths import BACKTEST_SETTINGS_DIR, TRAIN_SETTINGS_DIR
from deepm.backtest.signal_backtest import SignalBacktestJob
from deepm.backtest.signal_backtest_ensemble import EnsembleSignalBacktestJob
from deepm.data.dataset import get_transaction_costs
from deepm.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _validate_tickers_in_tcost(configs):
    """Check that all universe tickers appear in the transaction cost file."""
    tcost_file = configs.get("ticker_reference_file")
    if tcost_file is None:
        return

    ticker_ref = get_transaction_costs(tcost_file)
    ticker_mapping = configs.get("ticker_mapping")
    if ticker_mapping is not None:
        ticker_ref["ticker"] = ticker_ref["ticker"].map(ticker_mapping)

    universe = configs.get("universe", {})
    universe_tickers = set(universe.keys())
    if ticker_mapping is not None:
        universe_tickers = {ticker_mapping.get(t, t) for t in universe_tickers}

    tcost_tickers = set(ticker_ref["ticker"].dropna().unique())
    missing = universe_tickers - tcost_tickers

    if missing:
        logger.error(
            "The following tickers are in the backtest universe but missing from "
            "the transaction cost file '%s': %s",
            tcost_file,
            sorted(missing),
        )
        sys.exit(1)


def _resolve_tickers(configs, override_tickers=None):
    """Extract ticker list and group mapping from backtest config."""
    if override_tickers is not None:
        return override_tickers, {}

    universe = configs["universe"]
    ticker_list = list(universe.keys())

    has_mapping = "ticker_mapping" in configs
    ticker_mapping = configs.get("ticker_mapping", {})
    ticker_groups = {}
    for tkr in ticker_list:
        mapped = ticker_mapping.get(tkr) if has_mapping else tkr
        if mapped is not None:
            ticker_groups[mapped] = universe[tkr].get("group")

    return ticker_list, ticker_groups


def _load_train_config(model_cfg):
    """Load the training YAML referenced by a model config."""
    path = TRAIN_SETTINGS_DIR / f"{model_cfg['train_yaml']}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_single(args, configs):
    ticker_list, ticker_groups = _resolve_tickers(configs, args.tickers)
    model_cfg = configs["model"]

    job = SignalBacktestJob(
        args.name,
        ticker_list,
        args.start_date,
        data_parquet=configs["data_parquet"],
        save_dir=os.path.join(configs["save_path"], args.name),
        cfg_model=model_cfg,
        cfg_train=_load_train_config(model_cfg),
        transaction_cost_path=configs["ticker_reference_file"],
        variable_importance_mode=args.variable_importance_mode,
        output_signal_weights=args.output_signal_weights,
        end_date=args.end_date,
        ticker_mapping=configs.get("ticker_mapping", {}),
    )
    job.main(
        diagnostics=args.diagnostics and not args.output_signal_weights,
        ticker_groups=ticker_groups,
    )


def _run_ensemble(args, configs):
    job = EnsembleSignalBacktestJob(
        args.name,
        args.start_date,
        data_parquet=configs["data_parquet"],
        save_dir=os.path.join(configs["save_path"], args.name),
        list_models=configs["models"],
        transaction_cost_path=configs["ticker_reference_file"],
        variable_importance_mode=args.variable_importance_mode,
        output_signal_weights=args.output_signal_weights,
        end_date=args.end_date,
        smooth_signal_span=configs.get("smooth_signal_span"),
    )
    job.main(
        diagnostics=args.diagnostics and not args.output_signal_weights,
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Run backtests")
    parser.add_argument("--name", required=True)
    parser.add_argument("--ensemble", action="store_true",
                        help="Run ensemble backtest (config must contain 'models' list)")
    parser.add_argument("--start_date", default="2010-01-01")
    parser.add_argument("--end_date", default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--variable_importance_mode", action="store_true")
    parser.add_argument("--output_signal_weights", action="store_true")
    return parser.parse_known_args()[0]


def main():
    args = _parse_args()

    if args.variable_importance_mode:
        raise NotImplementedError("--variable_importance_mode is not yet supported")
    if args.output_signal_weights:
        raise NotImplementedError("--output_signal_weights is not yet supported")

    t0 = time.time()

    with open(BACKTEST_SETTINGS_DIR / f"{args.name}.yaml", encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    _validate_tickers_in_tcost(configs)

    if args.ensemble:
        _run_ensemble(args, configs)
    else:
        _run_single(args, configs)

    logger.info("Done! Elapsed: %.2f seconds.", time.time() - t0)


if __name__ == "__main__":
    main()
