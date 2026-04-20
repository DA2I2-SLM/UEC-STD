# TimeSeriesECM

TimeSeriesECM augments a forecasting backbone with an **Error Correction Module (ECM)** that learns to predict the residual between the backbone forecast and the ground truth, then adds a scaled version of that residual back to improve accuracy on long-term time series forecasting.

Built on top of [Time-Series-Library](https://github.com/thuml/Time-Series-Library).

---

## Installation

```bash
conda create -n timeecm python=3.8
conda activate timeecm
pip install -r requirements.txt
```

## Supported Datasets

Scripts are provided for the following benchmark datasets:

| Dataset | Script directory |
|---|---|
| ETT (ETTh1, ETTh2, ETTm1, ETTm2) | `scripts/long_term_forecast/ETT_script/` |
| ECL (Electricity) | `scripts/long_term_forecast/ECL_script/` |
| Weather | `scripts/long_term_forecast/Weather_script/` |
| Traffic | `scripts/long_term_forecast/Traffic_script/` |
| Exchange Rate | `scripts/long_term_forecast/Exchange_script/` |
| Solar Energy | `scripts/long_term_forecast/Solar_script/` |
| ILI | `scripts/long_term_forecast/ILI_script/` |

Use the standard benchmark dataset layout expected by Time-Series-Library. Point `--root_path` to the dataset directory and `--data_path` to the target CSV file.

## Supported Backbone Models

The ECM module supports the following backbone architectures:

iTransformer, Mamba, PatchTST, TimeBridge, TimeMixer, TimesNet, TimeXer, TSMixer

Other models are available as plain backbone-only baselines (no ECM) via the non-`_test` scripts.

---

## Quick Start

### Step 1 — Train the backbone model

```bash
bash ./scripts/long_term_forecast/<Dataset_script>/<Model>_<Dataset>.sh
```

Example:

```bash
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1.sh
bash ./scripts/long_term_forecast/Weather_script/TimeMixer.sh
bash ./scripts/long_term_forecast/ECL_script/iTransformer.sh
```

### Step 2 — Train and evaluate the ECM branch

```bash
bash ./scripts/long_term_forecast/<Dataset_script>/<Model>_<Dataset>_test.sh <mode> <use_ar> <pred_len> <errcor_coef> <ecm_model> <season_coef> <trend_coef>
```

Example:

```bash
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_test.sh 4 1 336 1 linear 0 0
bash ./scripts/long_term_forecast/Weather_script/TimeMixer_test.sh 4 1 336 1 linear 0 0
```

---

## Script Arguments

The `*_test.sh` scripts accept positional arguments in this order:

| Position | Argument | Description |
|---|---|---|
| 1 | `mode` | Run mode (see table below) |
| 2 | `use_ar` | Autoregressive decoding flag (`1` = enabled) |
| 3 | `pred_len` | Forecast horizon length |
| 4 | `errcor_coef` | Error-correction coefficient (see below) |
| 5 | `ecm_model` | ECM model name (e.g. `linear`) |
| 6 | `season_coef` | Seasonal component coefficient |
| 7 | `trend_coef` | Trend component coefficient |

### Run modes

| Mode | Description |
|---|---|
| `1` | Train backbone, then evaluate on test set |
| `0` | Evaluate backbone on test set (no training) |
| `2` | Autoregressive rolling-window inference with backbone only |
| `3` | Train ECM from backbone-generated features |
| `4` | Evaluate ECM |
| `5` | Train ECM with seasonal and trend components |
| `6` | Evaluate ECM with seasonal and trend components |

---

## Error-Correction Coefficient

The `errcor_coef` controls how strongly the ECM residual is blended into the backbone forecast:

- `0` — disables error correction; returns the backbone forecast unchanged
- `> 0` — scales the ECM-predicted residual added to the backbone forecast

### Automatic coefficient selection

Instead of choosing the coefficient manually, the code searches for the best value on a held-out validation split during modes `3` and `5`. Pass a comma-separated list of candidate values via `--error_flags` (default: `0.01,0.03,0.05,0.07,0.1,0.3,0.5,0.7,1`). Each candidate is evaluated against both MSE and MAE criteria, and the best-performing coefficient is encoded in the saved ECM checkpoint name.

---

## CLI Reference

Key flags defined in [run.py](run.py):

| Flag | Description |
|---|---|
| `--task_name` | Task type (e.g. `long_term_forecast`) |
| `--is_training` | `1` for train-then-test, `0` for inference only |
| `--model_id`, `--model` | Experiment ID and backbone model name |
| `--data`, `--root_path`, `--data_path` | Dataset name and file paths |
| `--seq_len`, `--label_len`, `--pred_len` | Input, label, and forecast lengths |
| `--use_ar` | Enable autoregressive decoding |
| `--checkpoints` | Directory for saving model checkpoints |
| `--alpha` | Temporal smoothing factor between consecutive autoregressive steps |
| `--errcor_coef` | Runtime error-correction coefficient |
| `--ecm_model` | ECM model architecture (e.g. `linear`, `CNN`, `RNN`) |
| `--err_h` | Hidden dimension of the ECM model |
| `--kernel_size` | Kernel size for moving-average decomposition (seasonal/trend ECM) |
| `--season_coef`, `--trend_coef` | Loss weights for seasonal and trend components in ECM training |
| `--error_flags` | Comma-separated candidates for automatic coefficient selection |
| `--wandb` | Enable Weights & Biases logging (`True`/`False`) |
| `--use_dtw` | Include DTW metric in evaluation |

---

## Outputs

| Output | Description |
|---|---|
| `result_long_term_forecast.txt` | Text summary of all evaluation runs |
| `metrics.npy` | Saved metric arrays (MAE, MSE, RMSE, MAPE, MSPE) |
| `pred.npy` / `true.npy` | Predictions and ground truth |
| PDF plots | Per-setting figures saved under `./scratch/` |
| ECM results (modes 3–4) | Written to `./scratch/infer_results/<setting>-<ecm>/` |
| ECM results (modes 5–6) | Written to `./scratch/infer_results_seasonal_<s>_trend_<t>_<k>/<setting>-<ecm>/` |

---

## Repository Structure

```
models/          # Backbone model implementations
layers/          # Shared layer components
exp/             # Experiment runners (training, testing, ECM inference)
data_provider/   # Dataset loading utilities
scripts/
  long_term_forecast/
    ETT_script/          # ETTh1/h2/m1/m2 scripts
    ECL_script/          # Electricity dataset scripts
    Weather_script/      # Weather dataset scripts
    Traffic_script/      # Traffic dataset scripts
    Exchange_script/     # Exchange rate dataset scripts
    Solar_script/        # Solar energy dataset scripts
    ILI_script/          # ILI dataset scripts
run.py           # Main entry point
```
