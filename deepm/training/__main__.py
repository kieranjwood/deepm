"""CLI entry-point for hyperparameter tuning.

Usage::

    python -m deepm.training -r deepm-gat -a DeePM
"""

import argparse
import copy

from joblib import Parallel, delayed

from deepm.configs.load import (
    ARCHITECTURES,
    load_settings_for_architecture,
    load_sweep_settings,
)
from deepm.training.data_setup import (
    build_correlation_features,
    build_windows,
    compute_scalers,
    create_datasets,
    load_and_filter_data,
    validate_tickers_in_tcost,
)
from deepm.training.hp_tuner import Tuner
from deepm.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run(
    settings: dict,
    data_parquet: str,
    sweep_settings: dict,
    architecture: str,
    valid_end_optional: int = None,
    end_date: str = None,
    filter_start_years: list[int] | None = None,
    num_workers: int = 1,
) -> None:
    """Run hyperparameter tuning across rolling train/test windows."""
    settings = copy.deepcopy(settings)

    data = load_and_filter_data(settings, data_parquet, end_date)
    validate_tickers_in_tcost(settings, data)
    corr_feat = build_correlation_features(settings, data)
    windows = build_windows(settings, filter_start_years)

    def run_window(window):
        train_start, test_start, test_end = window
        logger.info("Configuration: %s", settings["description"])
        logger.info("Test window: %s-%s", test_start, test_end)

        scalers = compute_scalers(data, settings, train_start, test_start)
        train_data, valid_data, test_data, train_extra_data, data_params = (
            create_datasets(
                settings, data, train_start, test_start, test_end,
                valid_end_optional, corr_feat,
            )
        )

        tuner = Tuner(
            {**settings, **scalers}, sweep_settings, architecture,
            train_data, valid_data, test_data, test_start, test_end,
            train_extra_data, data_params, valid_end_optional,
        )
        tuner.hyperparameter_optimisation()

    if num_workers > 1:
        Parallel(n_jobs=num_workers)(delayed(run_window)(w) for w in windows)
    else:
        for w in windows:
            run_window(w)


def _parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning")
    parser.add_argument("-r", "--run_file_name", default="pinnacle-gross-futs-and-fx",
                        help="Name of YAML file")
    parser.add_argument("-a", "--arch", default="DeePM", choices=ARCHITECTURES,
                        help="Architecture name")
    parser.add_argument("-veo", "--valid_end_optional", type=int, default=None,
                        help="Optionally end the valid period earlier")
    parser.add_argument("-ed", "--end_date", default=None,
                        help="Specify an end-date for backtest")
    parser.add_argument("-fsy", "--filter_start_years", type=int, nargs="+", default=None,
                        help="Filter start years")
    parser.add_argument("-n", "--num_workers", type=int, default=1,
                        help="Number of workers")
    return parser.parse_args()


def main():
    args = _parse_args()
    settings = load_settings_for_architecture(args.run_file_name, args.arch)
    sweep_settings = load_sweep_settings(settings["sweep_yaml"])

    run(
        settings,
        settings["data_parquet"],
        sweep_settings,
        args.arch,
        args.valid_end_optional,
        args.end_date,
        filter_start_years=args.filter_start_years,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
