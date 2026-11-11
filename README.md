# DeePM: Regime-Robust Deep Learning for Systematic Macro Portfolio Management

**[Paper](https://arxiv.org/abs/2601.05975)** | **[PDF](https://arxiv.org/pdf/2601.05975)**

Kieran Wood, Stephen J. Roberts, Stefan Zohren

All of our works are covered on my [website](https://kieranjwood.github.io/).

This repository contains the code to reproduce the experiments in the paper.

> We propose DeePM (Deep Portfolio Manager), a structured deep-learning macro portfolio manager trained end-to-end to maximize a robust, risk-adjusted utility. DeePM addresses three fundamental challenges in financial learning: (1) it resolves the asynchronous "ragged filtration" problem via a Directed Delay (Causal Sieve) mechanism that prioritizes causal impulse-response learning over information freshness; (2) it combats low signal-to-noise ratios via a Macroeconomic Graph Prior, regularizing cross-asset dependence according to economic first principles; and (3) it optimizes a distributionally robust objective where a smooth worst-window penalty serves as a differentiable proxy for Entropic Value-at-Risk (EVaR) — a window-robust utility encouraging strong performance in the most adverse historical subperiods. In large-scale backtests from 2010–2025 on 50 diversified futures with highly realistic transaction costs, DeePM attains net risk-adjusted returns that are roughly twice those of classical trend-following strategies and passive benchmarks, solely using daily closing prices. Furthermore, DeePM improves upon the state-of-the-art Momentum Transformer architecture by roughly fifty percent. The model demonstrates structural resilience across the 2010s "CTA Winter" and the post-2020 volatility regime shift, maintaining consistent performance through the pandemic, inflation shocks, and the subsequent higher-for-longer environment. Ablation studies confirm that strictly lagged cross-sectional attention, graph prior, principled treatment of transaction costs, and robust minimax optimization are the primary drivers of this generalization capability.

If you use this code, please cite:

```bibtex
@article{wood2026deepm,
  title={DeePM: Regime-Robust Deep Learning for Systematic Macro Portfolio Management},
  author={Wood, Kieran and Roberts, Stephen J and Zohren, Stefan},
  journal={arXiv preprint arXiv:2601.05975},
  year={2026}
}
```

## Repository Structure

```
deepm/                      # Core library
  _paths.py                 #   Project-wide path constants
  models/                   #   Neural network architectures
    base.py                 #     DeepMomentumNetwork base class + Sharpe loss
    deepm.py                #     SpatioTemporalGraphTransformer (main model)
    deepm_layers.py         #     Reusable building blocks (GNN, VSN, backbone, etc.)
    momentum_transformer.py #     Momentum Transformer baseline
    lstm.py                 #     LSTM baselines
    common.py               #     Shared building blocks (GRN, GateAddNorm, etc.)
  data/                     #   Data loading and feature engineering
    dataset.py              #     PyTorch datasets + transaction cost loading
    build_features.py       #     Feature computation (returns, vol, MACD, z-scores)
  training/                 #   Training pipeline
    __main__.py             #     CLI entry point (python -m deepm.training)
    train.py                #     Training loop + validation
    hp_tuner.py             #     Hyperparameter sweep via wandb
    data_setup.py           #     Data loading, filtering, and dataset construction
    grad_accum.py           #     Two-pass exact gradient accumulation for Sharpe loss
  backtest/                 #   Backtesting framework
    __main__.py             #     CLI entry point (python -m deepm.backtest)
    signal_backtest.py      #     Single-model signal generation + diagnostics
    signal_backtest_ensemble.py  # Ensemble backtesting
    metrics.py              #     Performance metrics (Sharpe, Calmar, HAC t-stat)
    models/                 #     Backtest model wrappers
      traditional.py        #       TSMOM, MACD, MVO, ERC, Risk-Managed baselines
      deep_momentum.py      #       DL model inference wrapper
  configs/                  #   Python config loaders
    load.py                 #     YAML config loading
    settings.py             #     Global settings (WANDB_ENTITY)
  utils/                    #   Utilities
    logging_utils.py        #     Logger setup
    metrics.py              #     Sharpe ratio, Calmar ratio

configs/                    # YAML configuration files
  train_settings/           #   Training configs
  backtest_settings/        #   Backtest configs
  sweep_settings/           #   Hyperparameter sweep configs
  tcost/                    #   Transaction cost data

scripts/                    # Reproduction scripts
  reproduce.sh              #   End-to-end reproduction pipeline
  smoke_test.sh             #   Quick end-to-end verification (not for real results)
  build_graph.py            #   Macro adjacency matrix generation for GNN
  prepare_features.py       #   Feature preparation from raw data
  generate_smoke_configs.py #   Auto-generate smoke-test training/backtest configs
  aggregate_metrics.py      #   Aggregate backtest metrics into tables and CSVs
```

## Setup

**Requirements:** Python >= 3.10, CUDA-capable GPU recommended.
All experiments in the paper were performed on an NVIDIA GeForce RTX 4090 (24 GB).

```bash
# recommended: create a virtual environment with uv (https://docs.astral.sh/uv/)
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .

# alternatively, with plain pip
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Weights & Biases (wandb)

Training uses [wandb](https://wandb.ai) for experiment tracking and hyperparameter sweeps.

**First-time setup:**

1. Create a free account at [wandb.ai](https://wandb.ai).
2. Create a project called **`dmn`** in your wandb workspace (this is the project name referenced by all training configs).
3. Set your credentials:

```bash
export WANDB_ENTITY="your-wandb-username"
export WANDB_API_KEY="your-api-key"       # or run `wandb login`
```

To run training without a wandb account, use offline mode:

```bash
export WANDB_MODE=offline
```

wandb sweeps need the wandb server, so in offline (or `disabled`) mode — or with `local_sweep: true` in the
training config — the hyperparameter search runs as an equivalent **local random search** over the same sweep
file. Trial `i` uses seed `local_sweep_seed + i` (default 0) for both the hyperparameter draw and model
initialisation. Each finished trial is written to `local_trials/local-<i>.json` in the run directory, so an
interrupted search resumes where it stopped; a trial that raises is recorded and skipped
(`DEEPM_RETRY_FAILED=1` retries it).

To run several trials in parallel on one GPU, start `n` processes with `DEEPM_SHARD=k/n` (`k = 0..n-1`), then run
once without it to aggregate `all_runs.csv` / `best_runs.csv`. `scripts/run_sharded.sh <train_yaml> <arch> <n>`
does both.

### Device

Training and inference use CUDA if available, otherwise Apple Silicon (MPS), otherwise CPU. Set
`DEEPM_DEVICE=cpu|mps|cuda:0` to override.

## Data

### Raw data (`data/data_dec25.parquet`)

The raw data is not included in this repository due to data provider licensing
restrictions. You will need to source equivalent daily closing prices yourself
(e.g. from Bloomberg, Refinitiv, or a futures data vendor).

The pipeline expects a **wide-format price panel** saved as a Parquet file:

- **Index:** `DatetimeIndex` of trading dates (business days, 1990-01-02 to 2025-12-31)
- **Columns:** 50 short ticker codes (see table below)
- **Values:** Daily closing (settlement) prices for continuous front-month futures contracts

The column tickers map to the following 50 assets:

| Group | Ticker | Asset |
|-------|--------|-------|
| **Equities** | `EN` | Nasdaq 100 |
| | `ER` | Russell 2000 |
| | `ES` | S&P 500 |
| | `YM` | Dow Jones |
| | `LX` | FTSE 100 |
| | `CA` | CAC 40 |
| | `XU` | EuroStoxx 50 |
| | `NK` | Nikkei 225 |
| | `HS` | Hang Seng |
| **Rates** | `FB` | US 5-Year Note |
| | `TU` | US 2-Year Note |
| | `TY` | US 10-Year Note |
| | `US` | US 30-Year Bond |
| | `DT` | German Bund (10yr) |
| | `UB` | German Bobl (5yr) |
| | `UZ` | German Schatz (2yr) |
| | `GS` | UK Gilt |
| | `CB` | Canada 10-Year |
| **Energy** | `BC` | Brent Crude |
| | `BG` | Gasoil |
| | `ZB` | RBOB Gasoline |
| | `ZN` | Natural Gas |
| | `ZU` | WTI Crude |
| **Metals** | `ZA` | Palladium |
| | `ZG` | Gold |
| | `ZI` | Silver |
| | `ZP` | Platinum |
| | `ZK` | Copper |
| **Agriculture** | `KW` | KC Wheat |
| | `ZC` | Corn |
| | `ZL` | Soybean Oil |
| | `ZM` | Soybean Meal |
| | `ZS` | Soybeans |
| | `ZW` | Chicago Wheat |
| | `CC` | Cocoa |
| | `CT` | Cotton |
| | `JO` | Orange Juice |
| | `KC` | Coffee |
| | `SB` | Sugar |
| **Livestock** | `ZF` | Feeder Cattle |
| | `ZT` | Live Cattle |
| | `ZZ` | Lean Hogs |
| **FX** | `AN` | AUD/USD |
| | `BN` | GBP/USD |
| | `CN` | CAD/USD |
| | `FN` | EUR/USD |
| | `JN` | JPY/USD |
| | `SN` | CHF/USD |
| | `MP` | MXN/USD |
| | `DX` | US Dollar Index |

**Example (first few rows and columns):**

```
              AN       BC       BN       CA  ...
1990-01-02  0.7745  20.65    1.6312   3245  ...
1990-01-03  0.7738  20.80    1.6298   3240  ...
1990-01-04  0.7721  20.72    1.6285   3238  ...
```

Missing values (e.g. for contracts that started trading after 1990) should be left
as `NaN`; the feature pipeline handles them automatically.

### Ticker naming conventions

The codebase uses two ticker conventions:

- **Short codes** (`EN`, `ES`, `ZG`, `AN`, etc.) — used in the raw data columns,
  the feature parquet, the transaction cost file (`configs/tcost/`), and the
  `ticker_subset` list in training configs.
- **Bloomberg tickers** (`NQ1 Index`, `ES1 Index`, `GC1 Comdty`, `AD1 Curncy`,
  etc.) — used by the GNN adjacency matrix (`macro_adjacency_normalized.csv`) and
  internally by the DeePM model during training and inference.

The `ticker_mapping` dictionary in each training and backtest config bridges the
two conventions, mapping every short code to its Bloomberg equivalent:

```yaml
ticker_mapping:
  EN: "NQ1 Index"
  ES: "ES1 Index"
  ZG: "GC1 Comdty"
  AN: "AD1 Curncy"
  # ... (all 50 assets)
```

When `ticker_mapping` is present, the pipeline applies it at data-load time so
that all downstream components (datasets, models, diagnostics) operate on
Bloomberg tickers. This is required for the GNN, which looks up assets by their
Bloomberg ticker in the adjacency matrix.

> **If you are not using the GNN** (e.g. the `deepm-independent` or
> `MOM_TRANS` configs), `ticker_mapping` can be omitted — the pipeline will
> use the short codes throughout.

### Feature preparation

The feature preparation step (`scripts/prepare_features.py`) converts the raw price
panel into a long-format DataFrame with all model inputs.

> **Note:** The pipeline computes a comprehensive feature set, but each training
> configuration uses only a subset specified in its `features` list. The main
> DeePM-GAT model uses six features: `r1d, r1m, r3m, r1y, z_price_21d,
> z_price_252d` (four normalised momentum returns + two price z-scores). The MACD
> ablation replaces the momentum returns with MACD signals. Refer to individual
> configs in `configs/train_settings/` for the exact feature set used by each model.

Available columns in the prepared feature file:

| Column | Description |
|--------|-------------|
| `ticker` | Asset identifier |
| `close` | Closing price |
| `target` | Next-day return (prediction target) |
| `r1d`, `r1w`, `r1m`, `r3m`, `r6m`, `r1y` | Normalized momentum features at various horizons |
| `macd_4_12`, `macd_8_24`, `macd_16_48`, `macd_32_96` | MACD signals |
| `z_price_*`, `z_ret_*` | Z-scored price/return features |
| `log_vol`, `log_vol_252d`, `z_log_vol_252d` | Volatility features |
| `skew_63d`, `kurt_63d` | Higher-moment features |
| `dd_252d`, `dd_63d`, `time_since_1y_high` | Drawdown features |
| `cs_pct_*`, `cs_z_*` | Cross-sectional rank/z-score features |
| `vs_factor` | Volatility scaling factor |

### Transaction costs (`configs/tcost/`)

The file `configs/tcost/futs_and_fx.csv` provides per-asset transaction cost
estimates used by the training loss and the backtest engine.  Each row
corresponds to one futures contract:

| Column | Description |
|--------|-------------|
| `ticker` | Short ticker code (matches the data and config files). |
| `description` | Human-readable contract name. |
| `bloomberg_ticker` | Bloomberg identifier. |
| `group` | Sector/asset-class label (e.g. `EQUITY_US`, `COMM_ENERGY`, `FX_G10`). |
| `open_utc`, `close_utc` | Trading session times (UTC). |
| `transaction_cost` | Half bid-ask spread in basis points — the primary value used by the pipeline. |

To use your own cost estimates, create a CSV with the same schema and point
`ticker_reference_file` in your configs to its filename (without path — the
loader looks in `configs/tcost/`).

> **Important:** Every ticker in your training config's `ticker_subset` and
> backtest config's `universe` must have a matching row in the transaction cost
> file. The pipeline validates this at startup and exits with an error listing
> any missing tickers. If you add new assets, add corresponding rows to the
> tcost CSV before training or backtesting.

## Reproduction

### Quick Start

```bash
bash scripts/reproduce.sh                # run full pipeline (train + backtest + tables)
bash scripts/reproduce.sh --step 2       # run a single step (see below)
bash scripts/reproduce.sh --baselines    # traditional baselines only (no GPU / no training)
bash scripts/smoke_test.sh               # fast end-to-end check (5 HP trials, 5 seeds, 1 window)
bash scripts/smoke_test.sh --baselines   # smoke baselines only
```

> **Note:** The smoke test uses drastically reduced settings (5 HP trials, 5 seeds,
> single 2020–2025 test window) and exists **only to verify the pipeline runs
> end-to-end without errors**.  Smoke metrics are not meaningful and should not be
> compared to the paper results.  Use `scripts/reproduce.sh` for publishable numbers.

The pipeline steps are:

| Step | What it does |
|------|-------------|
| 1 | Generate macro adjacency matrix (`scripts/build_graph.py`) |
| 2 | Prepare features from raw prices (`scripts/prepare_features.py`) |
| 3 | Train all DL models — DeePM variants + Momentum Transformer (HP sweep via wandb) |
| 4 | Run backtests for all models including traditional baselines, and generate diagnostics |
| 5 | Aggregate metrics into paper-style tables and save CSV |

### Step-by-step

**1. Generate adjacency matrix**

```bash
python scripts/build_graph.py
```

Produces `macro_adjacency_normalized.csv` encoding macro-economic relationships
between assets. The matrix is indexed by Bloomberg-style tickers (e.g.
`ES1 Index`, `GC1 Comdty`) and is loaded by the DeePM model at training time.

> **Custom universes:** If you change the asset universe, you must update
> `scripts/build_graph.py` to reflect your new assets. Specifically:
>
> - **`ASSET_LIST`** — the ordered list of Bloomberg tickers (defines row/column
>   order of the adjacency matrix).
> - **`MACRO_GROUPS`** — assigns each asset to a macro group (e.g. `EQUITY_US`,
>   `COMM_ENERGY`). Assets within the same group are connected automatically.
> - **`REGIONAL_LINKS`** — defines cross-asset-class connections within regions
>   (e.g. US equities ↔ US Treasuries ↔ USD).
> - **Macro story rules** (lines 113–160) — encode economic relationships like
>   risk-on, inflation, and safe-haven links. Add or remove links as appropriate
>   for your universe.
>
> After editing, re-run `python scripts/build_graph.py` to regenerate the
> adjacency CSV. You can use `--verify` to spot-check specific edges.

**2. Prepare features**

```bash
python scripts/prepare_features.py --input data/data_dec25.parquet
```

**3. Train models**

```bash
# Main model (DeePM-GAT)
python -m deepm.training -r "deepm-gat" -a DeePM

# Temporal baseline (Momentum Transformer)
python -m deepm.training -r "temporal-baseline" -a AdvancedTemporalBaseline
```

**4. Run backtests**

```bash
# DeePM-GAT
python -m deepm.backtest --name "bt-deepm-gat" --diagnostics

# Traditional baselines
python -m deepm.backtest --name "bt-baseline-tsmom" --diagnostics
```

### Output structure

After running the pipeline, the following directories are created:

```
models_torch/
  <train_config>/                 # e.g. deepm-gat/
    <architecture>/               #   e.g. DeePM/
      <test_start_year>/          #     e.g. 2010/, 2015/, 2020/
        all_runs.csv              #       All HP trial results (ranked by val Sharpe)
        best_runs.csv             #       Top-K seeds retained for backtesting
        best_runs_mean.json       #       Mean metrics across top-K seeds
        models/                   #       Saved model checkpoints (.pt)
        settings/                 #       Per-seed HP configs (.json)
        data-params/              #       Dataset scalers (.pkl)
        returns/                  #       Per-seed train/valid returns

backtest_results/
  <backtest_config>/              # e.g. bt-deepm-gat/
    <ticker>.parquet              #   Per-asset daily predictions/positions
  *_metrics.csv                   #   Aggregated metrics tables (from Step 5)

backtest_diagnostics/
  <backtest_config>/              # e.g. bt-deepm-gat/
    metrics.json                  #   Sharpe, Calmar, CAGR, MaxDD, etc.
    pnl_cumulative.png            #   Cumulative PnL plot
    pnl_curves.csv                #   Daily PnL time series
    annual_sharpe.png             #   Year-by-year Sharpe bar chart
    group_gross_pnl.png           #   Per-group cumulative gross PnL
    group_net_pnl.png             #   Per-group cumulative net PnL
```

## Training Configuration Reference

Each YAML file in `configs/train_settings/` defines the full specification for a
training run.  Settings fall into five categories: **architecture**, **loss /
optimisation**, **data**, **training loop**, and **asset universe**.  Hyperparameters
marked *(swept)* are overridden by the companion sweep config referenced via
`sweep_yaml`.

### General

| Setting | Type | Description |
|---------|------|-------------|
| `sweep_yaml` | str | Name of the sweep config in `configs/sweep_settings/` (defines the HP grid). |
| `save_directory` | str | Root directory for model checkpoints (default `models_torch`). |
| `project` | str | Wandb project name. |
| `test_run` | bool | If `true`, run a tiny sanity-check loop instead of full training. |

### Architecture

| Setting | Type | Description |
|---------|------|-------------|
| `use_lagged_cross_section` | bool | Enable the cross-sectional (multi-asset) dataset with a one-step causal lag between the key/value assets and the query asset (the "Directed Delay" / Causal Sieve mechanism from the paper). |
| `use_staggered_close_kv` | bool | Use a staggered-close key/value alignment (experimental variant of the causal lag). Requires `use_lagged_cross_section: true`. |
| `use_xatt` | bool | Enable cross-asset multi-head attention. |
| `use_gnn` | bool | Enable the graph neural network (GNN) layer that encodes macro-economic relationships. |
| `use_rezero_attn` | bool | Apply ReZero initialisation to the attention residual connections (init scale = `rezero_init`). |
| `use_rezero_gnn` | bool | Apply ReZero initialisation to the GNN residual connections. |
| `rezero_init` | float | Initial scalar for ReZero gates (default `0.01`). |
| `gat_concat_heads` | bool | Concatenate (rather than average) GAT attention heads. |
| `use_single_head_spatial` | bool | Use a single attention head in the spatial (GNN) layer regardless of `num_heads`. |
| `gnn_weighted_edges` | bool | Allow the GNN to learn edge weights on the adjacency matrix. |
| `no_bias_in_prediction_head` | bool | Remove bias from the final prediction linear layer. |
| `prediction_head_init` | str | Initialisation mode for the prediction head: `"default"` (Kaiming), `"zero"`, or `"near_zero"`. |
| `prediction_head_std` | float | Standard deviation when using `"default"` init (default `0.001`). |
| `use_static_ticker` | bool | Provide a learnable per-asset embedding to the model (used in the temporal baseline). |

### Loss and Optimisation

| Setting | Type | Description |
|---------|------|-------------|
| `optimise_loss_function` | int | Loss function selector. `0` = differentiable Sharpe ratio (default for all configs in this repo). |
| `use_joint_pooled_softmin_sharpe` | bool | Enable the joint objective: pooled Sharpe + λ × SoftMin window Sharpe (the robust minimax proxy from the paper). |
| `joint_softmin_lambda` | float | λ weight on the SoftMin penalty in the joint objective. |
| `softmin_sharpe_beta` | float | β temperature for the SoftMin operator (larger = closer to hard min). |
| `softmin_sharpe_center` | bool | Subtract a log-normalisation constant from the SoftMin (no gradient effect). |
| `use_softmin_sharpe` | bool | Use pure SoftMin Sharpe (without the pooled term). Mutually exclusive with `use_joint_pooled_softmin_sharpe`. |
| `use_adamw` | bool | Use AdamW optimiser (vs. Adam). |
| `alpha_lr_mult` | float | Learning rate multiplier for ReZero α parameters (default `100.0`). |
| `grad_accum_hidden_threshold` | int | Trigger two-pass gradient accumulation when `hidden_dim` exceeds this threshold. Splits the batch into `hidden_dim // threshold` chunks for memory efficiency while preserving exact gradients for the coupled Sharpe objective. |
| `grad_accum_ratio` | int | Explicit number of chunks for gradient accumulation (default `1` = disabled). Overridden by `grad_accum_hidden_threshold` when set. |

#### Two-Pass Exact Gradient Accumulation

Standard gradient accumulation (summing gradients across micro-batches) does not
produce correct gradients for the Sharpe ratio loss because the loss couples all
returns in the batch through shared mean and variance statistics. Naively
accumulating gradients across chunks treats each chunk as independent, introducing
approximation error.

The two-pass scheme in `deepm/training/grad_accum.py` (Appendix B of the
[paper](https://arxiv.org/abs/2601.05975)) solves this exactly:

1. **Pass 1 (no grad):** Stream through all chunks, accumulating the pooled sum
   and sum-of-squares of captured returns across the full batch. If the softmin or
   joint objective is active, per-sample Sharpe ratios and softmin weights are also
   computed. RNG states are saved so that dropout masks can be replayed.

2. **Pass 2 (with grad):** Replay each chunk with the saved RNG states. Using the
   global statistics from Pass 1, analytically derived gradients of the pooled
   Sharpe (and optionally the softmin-weighted per-sample Sharpe) are applied via a
   surrogate backward pass (`(returns * grad).sum().backward()`), producing exact
   parameter gradients identical to what a single full-batch forward/backward would
   yield.

This is triggered automatically when `grad_accum_hidden_threshold` is set: the
batch is split into `hidden_dim // threshold` chunks. For example, with
`hidden_dim: 128` and `grad_accum_hidden_threshold: 64`, the batch is processed in
2 chunks, halving peak memory while preserving exact gradients. All DeePM configs
in this repo use `grad_accum_hidden_threshold: 64`.

### Transaction Costs

| Setting | Type | Description |
|---------|------|-------------|
| `use_transaction_costs` | bool | Include transaction costs in the loss function (γ > 0 in the paper). |
| `tcost_inputs` | bool | Feed transaction cost estimates as additional model inputs. |
| `fixed_trans_cost_bp_loss` | float | Fixed transaction cost in basis points applied in the loss (0 = use asset-specific estimates from `configs/tcost/`). |
| `volscale_tc_loss` | bool | Scale transaction costs by volatility in the loss. |
| `turnover_regulariser_scaler` | float | Weight on the position-change (turnover) penalty. Corresponds to γ in the paper: 0 = zero cost, 0.5 = half cost (default), 1.0 = full cost. |

### Data

| Setting | Type | Description |
|---------|------|-------------|
| `data_parquet` | str | Filename of the prepared feature parquet in `data/`. |
| `ticker_reference_file` | str | Name of the ticker reference file in `configs/tcost/` (maps tickers to transaction costs). |
| `features` | list[str] | Ordered list of feature columns to use as model inputs (see feature table above). |
| `use_correlation_features` | bool | Compute and include pairwise correlation features. |
| `seq_len` | int | Sequence length (number of time steps per sample). |
| `pre_loss_steps` | int | Number of initial time steps excluded from the loss (burn-in for the model to form internal state before signals are evaluated). |

### Training Loop

| Setting | Type | Description |
|---------|------|-------------|
| `iterations` | int | Maximum training epochs per seed. |
| `random_search_max_iterations` | int | Number of HP configurations sampled by wandb sweep. |
| `top_n_seeds` | int | Number of top seeds (by validation Sharpe) retained for backtesting. |
| `early_stopping` | int | Stop training if no improvement for this many epochs. |
| `smooth_alpha` | float | EMA smoothing factor for the validation metric (0 < α < 1). Higher = more weight on recent epochs. |
| `min_delta` | float | Minimum improvement in smoothed validation metric to count as progress. |
| `val_burnin_steps` | int | Number of initial epochs to skip before early stopping begins. |
| `train_valid_split` | float | Fraction of the training window used for training (remainder is validation). |

### Asset Universe and Test Windows

| Setting | Type | Description |
|---------|------|-------------|
| `first_train_year` | int | Start year for the training data (inclusive). |
| `final_test_year` | int | End year for the last test window (inclusive). |
| `first_test_year` | int | Earliest test-window start year. |
| `test_start_years` | list[int] | List of test-window start years. The model is retrained for each window, with all data up to that year used for training/validation. |
| `ticker_subset` | list[str] | List of ticker codes to include (must match columns in the data parquet). |
| `ticker_mapping` | dict | Maps short ticker codes to Bloomberg-style identifiers (used for labelling only). |
| `target` | str | Name of the prediction target column. |
| `targets` | list[str] | List of target columns (currently single-element: `["target"]`). |
| `target_vol` | float | Annualised target volatility for position scaling in the backtest (in %). |
| `reference_cols` | list[str] | Non-feature columns carried through the dataset (`["date", "ticker"]`). |

## Backtest Configuration Reference

Each YAML file in `configs/backtest_settings/` defines a backtest run.  All configs
share three top-level keys (`save_path`, `ticker_reference_file`, `data_parquet`),
a `model` block that selects the strategy and its parameters, and a `universe`
block that maps tickers to asset-class groups for diagnostics.

### Top-level

| Setting | Type | Description |
|---------|------|-------------|
| `save_path` | str | Root directory for backtest prediction outputs (default `backtest_results`). Diagnostics are saved separately under `backtest_diagnostics/`. |
| `ticker_reference_file` | str | Ticker reference in `configs/tcost/` for transaction cost lookup. |
| `data_parquet` | str | Filename of the prepared feature parquet in `data/`. |

### Model block (`model:`)

Settings inside the `model:` block vary by strategy type.  The first four keys
are common to all configs:

| Setting | Type | Description |
|---------|------|-------------|
| `module` | str | Python module path for the model class (e.g. `deepm.backtest.models.deep_momentum` or `deepm.backtest.models.traditional`). |
| `class` | str | Class name within the module (e.g. `DeepMomentum`, `TSMOM`, `MVO`, `TrendERC`, `RiskManagedTrend`, `LongOnly`). |
| `train_yaml` | str | Name of the training config (without `.yaml`) — used to locate trained model checkpoints and to load data/universe settings. |
| `architecture` | str | Architecture selector passed to the model (e.g. `"DeePM"`, `"AdvancedTemporalBaseline"`). |

**Deep-learning models** (`class: DeepMomentum`) additionally accept:

| Setting | Type | Description |
|---------|------|-------------|
| `top_n_seeds` | int | Number of best seeds (by validation Sharpe) to ensemble at inference. |
| `seq_len` | int | Input sequence length for inference (should match training). |
| `pre_loss_steps` | int | Number of initial time steps to discard from predictions (model burn-in). |
| `batch_size` | int | Inference batch size. |
| `use_first_n_valid_seeds` | int | If set, use the first N valid seeds from the sweep ordering instead of the top N by metric. Used in seed-ablation experiments. |

**Traditional baselines** accept strategy-specific parameters:

| Setting | Type | Description |
|---------|------|-------------|
| `features` | list[str] | Feature columns for the signal (e.g. `["r1y"]` for TSMOM). |
| `signal` | str | Signal type: `"tsmom"` or `"macd"`. |
| `sign_features` | list[str] | Features whose sign determines position direction (ERC, Risk-Managed). |
| `mu_features` | list[str] | Features used as expected-return proxies (MVO). |
| `mu_transform` | str | Transform applied to mu features: `"sign"` (take sign only) or `"raw"`. |
| `macd_features` | list[str] | List of MACD signal columns (e.g. `["macd_8_24", "macd_16_48", "macd_32_96"]`). |
| `cov_lookback` | int | Rolling window (trading days) for covariance estimation (MVO, ERC, Risk-Managed). |
| `cov_estimator` | str | Covariance estimator: `"ledoit_wolf"` (shrinkage) or `"sample"`. |
| `ridge` | float | Ridge regularisation added to the covariance diagonal. |
| `rebalance_every` | int | Rebalance frequency in trading days (1 = daily). |
| `sigma_tgt` | float | Target portfolio volatility (annualised, as a fraction; 1.0 = 100%). |
| `max_gross_leverage` | float | Maximum gross leverage constraint. |
| `max_abs_pos` | float | Maximum absolute position per asset (caps individual weights). |
| `tcost_penalty` | float | Transaction cost penalty in the MVO optimiser (MVO-TP variant). |

### Universe and ticker mapping

| Setting | Type | Description |
|---------|------|-------------|
| `universe` | dict | Maps each ticker to its asset-class `group` (e.g. `Index`, `Bond`, `Comdty`, `Energy`, `FX`). Groups are used for per-group PnL diagnostics. |
| `ticker_mapping` | dict | Maps short ticker codes to Bloomberg-style identifiers (labelling only). |

## Sweep Configuration Reference

Sweep configs in `configs/sweep_settings/` are standard
[wandb sweep](https://docs.wandb.ai/guides/sweeps) YAML files.  Each training
config references a sweep via its `sweep_yaml` key.  The sweep defines the
hyperparameter search space; the training config fixes everything else.

Two sweep configs are provided:

| Sweep | Used by | Description |
|-------|---------|-------------|
| `sweep-deepm` | All DeePM variants and ablations | Smaller grid (fixed lr, fixed batch size). |
| `sweep-temporal` | Temporal baselines (Momentum Transformer) | Wider lr and batch size ranges. |

### Sweep settings

| Setting | Type | Description |
|---------|------|-------------|
| `method` | str | Search strategy: `"random"` (random search), `"grid"`, or `"bayes"`. |
| `metric.goal` | str | Optimisation direction: `"maximize"` or `"minimize"`. |
| `metric.name` | str | Metric to optimise (logged by training). `"valid_loss_best"` = best validation Sharpe. |

### Swept hyperparameters (`parameters:`)

| Parameter | `sweep-deepm` | `sweep-temporal` | Description |
|-----------|---------------|-------------------|-------------|
| `lr` | `[0.0001]` | `[0.0001, 0.0003]` | Learning rate. |
| `batch_size` | `[64]` | `[64, 128]` | Training batch size. |
| `weight_decay` | `[1e-5, 1e-4, 1e-3]` | `[1e-5, 1e-4, 1e-3]` | AdamW weight decay (L2 regularisation). |
| `max_gradient_norm` | `[0.5, 1.0]` | `[0.5, 1.0]` | Gradient clipping threshold. |
| `dropout` | `[0.3, 0.4, 0.5]` | `[0.1, 0.2, 0.3]` | Dropout probability. |
| `hidden_dim` | `[64, 128]` | `[64, 128]` | Hidden dimension of the transformer / GNN layers. |
| `num_heads` | `[2, 4]` | `[4]` | Number of multi-head attention heads. |

The total grid size is `random_search_max_iterations` samples (set in the training
config), and the top `top_n_seeds` configurations (by validation Sharpe) are
retained for backtesting.

## Config-to-Paper Mapping

### Training Configs (`configs/train_settings/`)

| Config | Architecture | Paper Reference |
|--------|-------------|-----------------|
| `deepm-gat` | DeePM (GAT) | Main model — Tables 1 & 2 |
| `deepm-gcn` | DeePM (GCN) | GCN (Isotropic) — Table 2 |
| `deepm-no-graph` | DeePM (no graph) | Cross-Attention Only — Table 2 |
| `deepm-graph-only` | DeePM (graph only) | Graph Only (No Cross-Attn) — Table 2 |
| `deepm-independent` | DeePM (independent) | Independent (No Structure) — Table 2 |
| `deepm-gat-flip` | DeePM (GAT, flip order) | Flip Graph/Cross-Attention — Table 2 |
| `deepm-gat-no-rezero` | DeePM (no ReZero) | No ReZero — Table 2 |
| `deepm-gat-macd` | DeePM (MACD features) | MACD Features — Table 2 |
| `deepm-gat-cascading` | DeePM (cascading lag) | Cascading Lag — Table 2 |
| `deepm-gat-pooled-only` | DeePM (pooled Sharpe) | No SoftMin (Pooled Only) — Table 2 |
| `deepm-gat-softmin-t1` | DeePM (SoftMin τ=1) | SoftMin τ=1 — Table 2 |
| `deepm-gat-softmin-t005` | DeePM (SoftMin τ=0.05) | SoftMin τ=0.05 — Table 2 |
| `deepm-gat-zero-cost` | DeePM (γ=0) | Zero Cost — Table 2 |
| `deepm-gat-full-cost` | DeePM (γ=1) | Full Cost — Table 2 |
| `temporal-baseline` | Momentum Transformer (γ=0.5) | Mom. Transformer — Table 1 |
| `temporal-baseline-zero-cost` | Momentum Transformer (γ=0) | Mom. Transformer (Zero Cost) — Table 1 |

### Backtest Configs (`configs/backtest_settings/`)

Each training config has a corresponding backtest config prefixed with `bt-` (full-period 2010–2025) and `bt-post2020-` (recent period 2020–2025).

| Config Pattern | Model | Paper Reference |
|--------|-------|-----------------|
| `bt-deepm-gat` | DeePM-GAT (K=25) | Main results — Table 1 |
| `bt-deepm-gat-{1,10,50,100}seed` | DeePM-GAT (seed ablation) | Seed sensitivity — Table 2 |
| `bt-deepm-gcn` | DeePM-GCN | GCN ablation — Table 2 |
| `bt-deepm-no-graph` | Cross-Attention Only | Architecture ablation — Table 2 |
| `bt-deepm-graph-only` | Graph Only | Architecture ablation — Table 2 |
| `bt-deepm-independent` | Independent | Architecture ablation — Table 2 |
| `bt-deepm-gat-flip` | Flip order | Architecture ablation — Table 2 |
| `bt-deepm-gat-no-rezero` | No ReZero | Architecture ablation — Table 2 |
| `bt-deepm-gat-macd` | MACD features | Feature ablation — Table 2 |
| `bt-deepm-gat-cascading` | Cascading lag | Feature ablation — Table 2 |
| `bt-deepm-gat-pooled-only` | No SoftMin | Loss ablation — Table 2 |
| `bt-deepm-gat-softmin-t1` | SoftMin τ=1 | Loss ablation — Table 2 |
| `bt-deepm-gat-softmin-t005` | SoftMin τ=0.05 | Loss ablation — Table 2 |
| `bt-deepm-gat-zero-cost` | Zero Cost (γ=0) | Cost ablation — Table 2 |
| `bt-deepm-gat-full-cost` | Full Cost (γ=1) | Cost ablation — Table 2 |
| `bt-temporal-baseline` | Momentum Transformer | Baseline — Table 1 |
| `bt-baseline-longonly` | Long Only | Baseline — Table 1 |
| `bt-baseline-tsmom` | TSMOM | Baseline — Table 1 |
| `bt-baseline-macd` | MACD | Baseline — Table 1 |
| `bt-baseline-macd-rm` | Risk-Managed MACD | Baseline — Table 1 |
| `bt-baseline-tsmom-mvo` | MVO Trend | Baseline — Table 1 |
| `bt-baseline-tsmom-mvo-tp` | MVO-TP Trend | Baseline — Table 1 |
| `bt-baseline-tsmom-erc` | ERC Trend | Baseline — Table 1 |
| `bt-baseline-tsmom-rm` | Risk-Managed Trend | Baseline — Table 1 |
| `bt-baseline-macd-mvo-tp` | MVO-TP MACD | Baseline — Table 1 |

## Adding a New Model

The codebase is designed so that new architectures can be added with minimal
boilerplate.  The steps below walk through the process end-to-end.

### 1. Implement the architecture

Create a new file in `deepm/models/` (e.g. `deepm/models/my_model.py`) and
subclass `DeepMomentumNetwork` from `deepm.models.base`:

```python
from deepm.models.base import DeepMomentumNetwork


class MyModel(DeepMomentumNetwork):
    """One-line description of the model."""

    def __init__(self, input_dim, num_tickers, **kwargs):
        super().__init__(input_dim, num_tickers, **kwargs)
        # Build layers here.  Key dimensions are available as:
        #   self.hidden_dim, self.dropout, self.seq_len, self.pre_loss_steps
        # Transaction cost inputs add self.extra_tcost_channels to the
        # feature dimension.  The prediction head (hidden_dim -> position)
        # is already defined in the base class.

    def forward_candidate_arch(self, target_x, target_tickers, **kwargs):
        """Core forward pass: features -> hidden representation.

        Args:
            target_x: Input tensor [B, T, F] (or [B, N, T, F] for cross-section).
            target_tickers: Ticker IDs [B] (or [B, N]).
            **kwargs: Extra inputs (vol_scaling, trans_cost, dates, etc.).

        Returns:
            Hidden tensor of shape [..., T, hidden_dim] which the base class
            feeds through the shared prediction head to produce positions.
        """
        ...

    def variable_importance(self, target_x, target_tickers, **kwargs):
        """Optional: return importance scores."""
        raise NotImplementedError
```

The base class handles everything downstream of the hidden representation:
the prediction head (tanh → positions in [-1, 1]), transaction cost
computation, Sharpe-ratio loss (pooled, softmin, or joint), asset dropout,
and position scaling.

If your model operates on the cross-sectional (multi-asset) batch layout,
set `self.is_cross_section = True` in `__init__`.

### 2. Register the architecture

**a) Add to the architecture list** in `deepm/configs/load.py`:

```python
ARCHITECTURES = [
    "LSTM",
    "LSTM_SIMPLE",
    "MOM_TRANS",
    "AdvancedTemporalBaseline",
    "DeePM",
    "MyModel",          # ← add here
]
```

If your model needs architecture-specific defaults (e.g. a different
`cross_section` setting), add them in `load_settings_for_architecture` in
the same file.

**b) Register the class** in `TrainDeepMomentumNetwork.ARCHITECTURE_CLASSES`
(`deepm/training/train.py`):

```python
ARCHITECTURE_CLASSES = {
    "LSTM": LstmBaseline,
    "LSTM_SIMPLE": LstmSimple,
    "DeePM": SpatioTemporalGraphTransformer,
    "AdvancedTemporalBaseline": AdvancedTemporalBaseline,
    "MOM_TRANS": MomentumTransformer,
    "MyModel": MyModel,          # ← add here
}
```

If your model needs extra kwargs injected at construction time (e.g. DeePM
requires `expected_ticker_order` and `ticker_close_rank` from the dataset),
add a `_inject_<name>_kwargs` helper and call it from `_load_architecture`:

```python
def _load_architecture(self, architecture, input_dim, num_tickers, optimise_loss_function, **kwargs):
    cls = self.ARCHITECTURE_CLASSES.get(architecture)
    ...
    if architecture == "MyModel":
        self._inject_mymodel_kwargs(kwargs)
    ...

def _inject_mymodel_kwargs(self, kwargs):
    """Populate any dataset-derived kwargs for MyModel."""
    ref_data = self.valid_data if self.valid_data is not None else self.test_data
    kwargs.setdefault("my_special_param", ref_data.some_attribute)
```

**c) Import** the class at the top of `deepm/training/train.py`:

```python
from deepm.models.my_model import MyModel
```

### 3. Create configuration files

**Training config** — copy an existing config (e.g. `configs/train_settings/deepm-gat.yaml`)
and modify architecture-specific settings.  The `sweep_yaml` key selects the
HP grid; you can reuse an existing sweep or create a new one in
`configs/sweep_settings/`.

**Backtest config** — create a corresponding file in `configs/backtest_settings/`:

```yaml
save_path: backtest_results
ticker_reference_file: futs_and_fx
data_parquet: "data_dec25.parquet"

model:
  module: "deepm.backtest.models.deep_momentum"
  class: "DeepMomentum"
  train_yaml: "my-model"
  architecture: "MyModel"
  top_n_seeds: 25
  seq_len: 84
  pre_loss_steps: 63
  batch_size: 8

universe:
  # ... (copy from an existing config)
```

### 4. Train and evaluate

```bash
# Train
python -m deepm.training -r "my-model" -a MyModel

# Backtest
python -m deepm.backtest --name "bt-my-model" --diagnostics
```

## License

MIT License. See [LICENSE](LICENSE).
