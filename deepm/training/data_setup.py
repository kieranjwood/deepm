"""Data loading, filtering, and dataset construction for the training pipeline."""

from __future__ import annotations

import sys

import pandas as pd
from colorama import Fore, Style

from deepm._paths import DATA_DIR
from deepm.data.dataset import (
    CrossSectionDataset,
    MomentumDataset,
    calc_trans_cost_scaler,
    calc_vs_factor_scaler,
    correlation_features,
    datetime_embedding_global,
    get_transaction_costs,
)
from deepm.utils.logging_utils import get_logger

logger = get_logger(__name__)


def validate_tickers_in_tcost(settings: dict, data: pd.DataFrame) -> None:
    """Check that all data tickers appear in the transaction cost file.

    Exits with a clear error message if any tickers are missing.
    """
    ticker_ref = get_transaction_costs(settings["ticker_reference_file"])
    if settings.get("ticker_mapping") is not None:
        ticker_ref["ticker"] = ticker_ref["ticker"].map(settings["ticker_mapping"])

    data_tickers = set(data.ticker.unique())
    tcost_tickers = set(ticker_ref["ticker"].dropna().unique())
    missing = data_tickers - tcost_tickers

    if missing:
        logger.error(
            "The following tickers are in the data but missing from the "
            "transaction cost file '%s': %s",
            settings["ticker_reference_file"],
            sorted(missing),
        )
        sys.exit(1)


def load_and_filter_data(
    settings: dict,
    data_parquet: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load the feature-enriched parquet and apply config-driven filters.

    Applies (in order): end-date truncation, test-mode subsetting (5 tickers),
    ticker-subset filtering, and ticker-mapping renaming.

    Args:
        settings: Training config dict.
        data_parquet: Filename of the raw parquet (e.g. ``data_dec25.parquet``).
            The loader prepends ``feats-`` and looks inside ``DATA_DIR``.
        end_date: Optional upper bound on the date index.

    Returns:
        Filtered DataFrame ready for windowed training.
    """
    feats_file_path = str(DATA_DIR / ("feats-" + data_parquet))

    if settings["test_run"]:
        logger.info(
            Fore.RED
            + "WARNING: In test mode and only five assets will be loaded"
            + Style.RESET_ALL
        )

    data = pd.read_parquet(feats_file_path)

    if end_date:
        data = data.loc[data.index <= end_date]

    if settings["test_run"]:
        data = data[data.ticker.isin(data.ticker.unique()[:5])]

    if settings.get("ticker_subset"):
        data = data[data.ticker.isin(settings["ticker_subset"])]

    if settings.get("ticker_mapping", None) is not None:
        data["ticker"] = data["ticker"].map(settings["ticker_mapping"])

    return data


def build_correlation_features(
    settings: dict,
    data: pd.DataFrame,
) -> dict | None:
    """Compute cross-sectional correlation features if enabled.

    Args:
        settings: Training config dict.
        data: Feature DataFrame (output of :func:`load_and_filter_data`).

    Returns:
        Correlation feature array, or ``None`` if disabled.
    """
    if not settings.get("use_correlation_features", False):
        return None

    if settings.get("use_principal_components", False):
        raise NotImplementedError(
            "PCA correlation features are not yet supported. "
            "Set use_principal_components: false in your config."
        )

    return correlation_features(
        data,
        settings["correlation_feature"],
        settings["correlation_span"],
        lag=1 if settings.get("use_lagged_cross_section", False) else 0,
    )


def build_windows(
    settings: dict,
    filter_start_years: list[int] | None = None,
) -> list[tuple[int, int, int]]:
    """Build the list of ``(train_start, test_start, test_end)`` windows.

    Windows are returned in reverse chronological order so that the most
    recent (and typically most informative) window trains first.

    Args:
        settings: Training config dict (needs ``first_train_year``,
            ``test_start_years``, ``final_test_year``).
        filter_start_years: If given, only keep windows whose
            ``test_start`` is in this list.

    Returns:
        List of ``(train_start, test_start, test_end)`` tuples.
    """
    windows = [
        (settings["first_train_year"], s, e)
        for s, e in zip(
            settings["test_start_years"],
            settings["test_start_years"][1:] + [settings["final_test_year"] + 1],
        )
        if (filter_start_years is None or s in filter_start_years)
    ][::-1]

    return windows


def compute_scalers(
    data: pd.DataFrame,
    settings: dict,
    train_start: int,
    test_start: int,
) -> dict:
    """Compute vol-scaling and transaction-cost scalers for a given window.

    Args:
        data: Feature DataFrame.
        settings: Training config dict.
        train_start: First year of the training period.
        test_start: First year of the test period (exclusive upper bound
            for the scaler calculation).

    Returns:
        Dict with keys ``vs_factor_scaler`` and ``trans_cost_scaler``.
    """
    data_ex_test = data[
        (pd.to_datetime(data.index).year >= train_start)
        & (pd.to_datetime(data.index).year < test_start)
    ]
    vs_factor_scaler = calc_vs_factor_scaler(data_ex_test)

    ticker_ref = get_transaction_costs(settings["ticker_reference_file"])
    if settings.get("ticker_mapping", None) is not None:
        ticker_ref["ticker"] = ticker_ref["ticker"].map(settings["ticker_mapping"])

    trans_cost_scaler = calc_trans_cost_scaler(
        ticker_ref[ticker_ref["ticker"].isin(data_ex_test.ticker.unique())]
    )
    return {
        "vs_factor_scaler": vs_factor_scaler,
        "trans_cost_scaler": trans_cost_scaler,
    }


def _resolve_dataset_class_and_extras(
    settings: dict,
    data: pd.DataFrame,
    corr_feat,
    train_start: int,
) -> tuple[type, dict]:
    """Pick the dataset class and build supplementary constructor kwargs.

    Args:
        settings: Training config dict.
        data: Feature DataFrame.
        corr_feat: Pre-computed correlation features (or ``None``).
        train_start: First year of the training period.

    Returns:
        ``(dataset_class, extra_kwargs)`` tuple.
    """
    dataset_class = CrossSectionDataset if settings["cross_section"] else MomentumDataset
    extra: dict = {}

    if settings.get("date_time_embedding", False):
        extra["date_time_embedding"] = datetime_embedding_global(
            train_start, data.index.year.max()
        )
        max_len = settings["datetime_embedding_global_max_length"]
        if len(extra["date_time_embedding"]) > max_len and not settings["local_time_embedding"]:
            raise ValueError(
                f"Date time embedding too long: {len(extra['date_time_embedding'])} "
                "- please train with a higher max length or reduce the number of years."
            )

    if corr_feat is not None:
        extra["corr_features"] = corr_feat

    return dataset_class, extra


def create_datasets(
    settings: dict,
    data: pd.DataFrame,
    train_start: int,
    test_start: int,
    test_end: int,
    valid_end_optional: int | None = None,
    corr_feat=None,
) -> tuple:
    """Create train, validation, test (and optional extra-train) datasets.

    Args:
        settings: Training config dict.
        data: Feature DataFrame.
        train_start: First year of the training period.
        test_start: First year of the test period.
        test_end: Exclusive upper-bound year for the test period.
        valid_end_optional: If set and earlier than ``test_start``, the
            validation set ends here and the gap is added as extra training
            data.
        corr_feat: Pre-computed correlation features (or ``None``).

    Returns:
        ``(train_data, valid_data, test_data, train_extra_data, data_params)``
    """
    dataset_class, extra = _resolve_dataset_class_and_extras(
        settings, data, corr_feat, train_start,
    )
    shift_valid = valid_end_optional and valid_end_optional < test_start
    end_yr = valid_end_optional if shift_valid else test_start

    valid_data = dataset_class(
        settings=settings, data=data, first_year=train_start, end_year=end_yr,
        drop_first_perc=settings["train_valid_split"],
        target_override=settings["valid_target_override"], **extra,
    )

    if settings["use_contexts"] and settings["cross_section"]:
        settings["num_context"] = valid_data.num_contexts

    train_data = dataset_class(
        settings=settings, data=data, first_year=train_start, end_year=end_yr,
        keep_first_perc=settings["train_valid_split"],
        tickers_dict=valid_data.tickers_dict,
        target_override=settings["train_target_override"], **extra,
    )

    train_extra_data = []
    if shift_valid:
        train_extra_data.append(dataset_class(
            settings=settings, data=data,
            first_year=valid_end_optional, end_year=test_start,
            tickers_dict=valid_data.tickers_dict,
            target_override=settings["train_target_override"], **extra,
        ))

    test_data = dataset_class(
        settings=settings, data=data, first_year=test_start, end_year=test_end,
        tickers_dict=valid_data.tickers_dict, test_set=True, **extra,
    )

    data_params: dict = {}
    if settings["use_contexts"] and not settings["cross_section"]:
        data_params["context_random_state"] = test_data.context_random_state

    return train_data, valid_data, test_data, train_extra_data, data_params
