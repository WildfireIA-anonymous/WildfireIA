# WildfireIA Code Release

This repository contains the anonymous code release for WildfireIA. The data are
released separately on Hugging Face:

<https://huggingface.co/datasets/WildfireIA/Anonymous-WildfireIA>

The Hugging Face repository contains **canonical benchmark tables only**. It
does not contain model-ready caches. After cloning this code repository, place
the Hugging Face canonical tables at the path shown below and regenerate caches
with `dataloader.py`.

## Quick Start

Clone this code repository and enter it:

```bash
git clone https://github.com/WildfireIA-anonymous/WildfireIA.git
cd WildfireIA
```

Install Python packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch GPU wheels depend on the local CUDA version. If the default `torch`
installation is not compatible with your system, install PyTorch following the
official PyTorch instructions, then rerun `pip install -r requirements.txt`.

Download the Hugging Face dataset into a temporary folder. The most portable
method is `huggingface_hub`, which downloads Git LFS files without requiring a
system `git-lfs` installation:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="WildfireIA/Anonymous-WildfireIA",
    repo_type="dataset",
    local_dir="hf_data",
)
PY
```

If `git-lfs` is already installed, this equivalent command also works:

```bash
git clone https://huggingface.co/datasets/WildfireIA/Anonymous-WildfireIA hf_data
```

Copy the canonical tables into the code repository:

```bash
mkdir -p data/canonical/raw_feature_tables
rsync -a hf_data/data/canonical/raw_feature_tables/ data/canonical/raw_feature_tables/
```

The expected path is:

```text
data/canonical/raw_feature_tables/
```

Generate Task 1 model-ready caches from the canonical tables:

```bash
python dataloader.py \
  --base_dir . \
  --canonical_dir data/canonical/raw_feature_tables \
  --output_dir data/cache/model_ready \
  --task ia_failure \
  --representation all \
  --weather_days 5 \
  --input_protocol all \
  --overwrite
```

Run one Task 1 model:

```bash
python train.py \
  --base_dir . \
  --task ia_failure \
  --experiment_type smoke \
  --representation tabular \
  --weather_days 5 \
  --input_protocol all \
  --model xgboost \
  --seed 553371 \
  --overwrite
```

The output is written to:

```text
experiments/ia_failure/smoke/tabular/weather5_all/xgboost_seed553371/
```

Important output files include:

```text
config.json
metrics.json
predictions_val.parquet
predictions_test.parquet
```

## Reproducing the Main Experiments

The official output directory format is:

```text
experiments/{task}/{experiment_type}/{representation}/weather{days}_{protocol}/{model}_seed{seed}/
```

For a full Task 1 run of XGBoost:

```bash
python train.py \
  --base_dir . \
  --task ia_failure \
  --experiment_type full \
  --representation tabular \
  --weather_days 5 \
  --input_protocol all \
  --model xgboost \
  --seed 553371 \
  --overwrite
```

For a spatial neural model:

```bash
python train.py \
  --base_dir . \
  --task ia_failure \
  --experiment_type full \
  --representation spatial \
  --weather_days 5 \
  --input_protocol all \
  --model swin_unet \
  --seed 553371 \
  --max_epochs 100 \
  --batch_size 64 \
  --early_stop_patience 15 \
  --sampling_strategy weighted \
  --standardize_channels \
  --overwrite
```

For Task 2, first generate Task 2 caches:

```bash
python dataloader.py \
  --base_dir . \
  --canonical_dir data/canonical/raw_feature_tables \
  --output_dir data/cache/model_ready \
  --task containment_time \
  --representation all \
  --weather_days 5 \
  --input_protocol all \
  --overwrite
```

Then run a Task 2 model:

```bash
python train.py \
  --base_dir . \
  --task containment_time \
  --experiment_type full \
  --representation tabular \
  --weather_days 5 \
  --input_protocol all \
  --model xgboost \
  --seed 553371 \
  --overwrite
```

## Summarizing Results

After full experiments finish:

```bash
python summarize_task1_full_all_seeds.py
python summarize_task2_full_all_seeds.py
```

Summary CSV and Markdown files are written under:

```text
results/
```

## What Each Script Does

- `pipeline.py`: optional raw-data canonicalization script. Reviewers do not
  need to run this when using the Hugging Face canonical tables.
- `dataloader.py`: converts canonical tables into model-ready caches.
- `train.py`: trains tabular, temporal, spatial, and spatiotemporal baselines.
- `summarize_*.py`: summarizes full-test and ablation experiment outputs.
- `scripts/prepare_hf_release.py`: prepares the anonymous Hugging Face data
  release.

## Supported Models

- Tabular: `logistic_regression`, `xgboost`, `mlp`
- Temporal: `gru`, `tcn`, `transformer`
- Spatial: `resnet18_unet`, `resnet50_unet`, `swin_unet`, `segformer`
- Spatiotemporal: `convlstm`, `convgru`, `predrnn_v2`, `utae`, `swinlstm`,
  `resnet3d`

## Notes for Anonymous Review

This repository contains code only. It does not include raw data, generated
caches, experiment logs, checkpoints, or paper source files. The canonical data
and Croissant metadata are hosted in the Hugging Face dataset repository linked
above.
