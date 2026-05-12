# WildfireIA Code Release

This repository contains anonymous code for the WildfireIA benchmark. The
benchmark evaluates whether public information available at wildfire discovery
time can predict initial attack failure and remaining time to containment.

The dataset is released separately on Hugging Face:

<https://huggingface.co/datasets/WildfireIA/Anonymous-WildfireIA>

The Hugging Face release contains canonical benchmark tables and Croissant
metadata. Model-ready caches are intentionally not stored in this code
repository; they can be regenerated from the canonical tables with
`dataloader.py`.

## Repository Contents

- `pipeline.py`: canonicalizes raw public-source inputs into benchmark tables.
- `dataloader.py`: converts canonical tables into model-ready caches.
- `train.py`: trains tabular, temporal, spatial, and spatiotemporal baselines.
- `summarize_*.py`: summarizes full-test and ablation experiment outputs.
- `scripts/prepare_hf_release.py`: prepares the anonymous Hugging Face data
  release.

## Installation

Create a Python environment with the packages in `requirements.txt`. GPU
training also requires a PyTorch build compatible with the local CUDA version.

```bash
pip install -r requirements.txt
```

For geospatial canonicalization from raw data, install optional system and
Python geospatial dependencies compatible with `geopandas`, `rasterio`,
`pyogrio`, and `pyproj`.

## Data Layout

After downloading or cloning the Hugging Face dataset, place or symlink its
canonical tables under:

```text
data/canonical/raw_feature_tables/
```

The expected canonical directory contains files such as:

```text
fire_events_natural_2016_2020.parquet
master_features_natural_2016_2020.parquet
event_static_patch_375m_natural_2016_2020.parquet/
event_weather_daily_patch_375m_natural_2016_2020.parquet/
```

## Generate Model-Ready Caches

Task 1 full-input caches:

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

Task 2 full-input caches:

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

## Train Baselines

Example Task 1 full-input run:

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

Example neural patch-model run:

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

Experiment outputs are written to:

```text
experiments/{task}/{experiment_type}/{representation}/weather{days}_{protocol}/{model}_seed{seed}/
```

## Supported Models

Task 1 and Task 2 use the same representation families:

- Tabular: `logistic_regression`, `xgboost`, `mlp`
- Temporal: `gru`, `tcn`, `transformer`
- Spatial: `resnet18_unet`, `resnet50_unet`, `swin_unet`, `segformer`
- Spatiotemporal: `convlstm`, `convgru`, `predrnn_v2`, `utae`, `swinlstm`,
  `resnet3d`

## Summaries

After experiments finish:

```bash
python summarize_task1_full_all_seeds.py
python summarize_task2_full_all_seeds.py
```

Summary tables are written under `results/`.

## Anonymous Review Notes

This repository is prepared for anonymous review. It should not contain author
names, affiliations, local machine paths, private data, experiment logs, or
large generated caches.

