#!/usr/bin/env python3
"""Prepare an anonymous Hugging Face dataset release for WildfireIA.

By default, the release contains canonical benchmark tables only. Model-ready
caches can be regenerated from the canonical tables; pass
``--include_task1_full_cache`` only when a larger cache-inclusive package is
needed. Large files are hardlinked when possible so the staging folder does not
duplicate local disk usage.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import date
from pathlib import Path

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None


DATASET_REPO_ID = "WildfireIA/Anonymous-WildfireIA"
HF_RESOLVE = f"https://huggingface.co/datasets/{DATASET_REPO_ID}/resolve/main"


CANONICAL_FILES = [
    "event_grid_375m_index_natural_2016_2020.parquet",
    "event_osm_patch_375m_natural_2016_2020.parquet",
    "event_patch_manifest_375m_natural.json",
    "event_static_patch_375m_natural_2016_2020.parquet",
    "event_viirs_patch_375m_D_natural_2016_2020.parquet",
    "event_weather_aggregate_patch_375m_natural_2016_2020.parquet",
    "event_weather_daily_patch_375m_natural_2016_2020.parquet",
    "feature_manifest_natural.json",
    "fire_events_natural_2016_2020.parquet",
    "gridmet_daily_event_features_natural_2016_2020.parquet",
    "gridmet_features_natural_2016_2020.parquet",
    "label_manifest_natural.json",
    "landfire_fuel_veg_features_natural_2016_2020.parquet",
    "master_features_natural_2016_2020.parquet",
    "osm_access_features_natural_2016_2020.parquet",
    "population_features_natural_2016_2020.parquet",
    "temporal_protocol_manifest_natural.json",
    "topography_features_natural_2016_2020.parquet",
    "viirs_features_natural_2016_2020.parquet",
]

TASK1_FULL_CACHE_DIRS = [
    "ia_failure/tabular/weather5_all",
    "ia_failure/temporal/weather5_all",
    "ia_failure/spatial/weather5_all",
    "ia_failure/spatiotemporal/weather5_all",
]

CODE_FILES = [
    "dataloader.py",
    "train.py",
    "summarize_task1_full_all_seeds.py",
    "summarize_task2_full_all_seeds.py",
]


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def copy_or_link_file(src: Path, dst: Path, hardlink: bool = True) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return
        dst.unlink()
    if hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def copy_or_link_tree(src: Path, dst: Path, hardlink: bool = True) -> None:
    if src.is_file():
        copy_or_link_file(src, dst, hardlink=hardlink)
        return
    for item in src.rglob("*"):
        if item.is_dir():
            continue
        copy_or_link_file(item, dst / item.relative_to(src), hardlink=hardlink)


def sanitized_code(text: str) -> str:
    replacements = {
        ".": ".",
        ".": ".",
        "anonymous": "anonymous",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parquet_summary(path: Path, release_root: Path) -> dict:
    summary = {"path": rel(path, release_root), "rows": None, "columns": None}
    if pq is None:
        return summary
    try:
        if path.is_dir():
            files = sorted(path.rglob("*.parquet"))
            rows = 0
            columns = None
            for file in files:
                meta = pq.ParquetFile(file).metadata
                rows += meta.num_rows
                if columns is None:
                    schema = pq.ParquetFile(file).schema_arrow
                    columns = schema.names
            summary["rows"] = rows
            summary["columns"] = columns
        else:
            pf = pq.ParquetFile(path)
            summary["rows"] = pf.metadata.num_rows
            summary["columns"] = pf.schema_arrow.names
    except Exception as exc:
        summary["error"] = str(exc)
    return summary


def build_readme(include_task1_full_cache: bool) -> str:
    cache_bullet = (
        "- `data/model_ready/ia_failure/`: Task 1 full-input model-ready caches for\n"
        "  tabular, temporal, spatial, and spatiotemporal models.\n"
        if include_task1_full_cache
        else "- Model-ready caches are not included in this compact release; regenerate\n"
        "  them deterministically from the canonical tables with `code/dataloader.py`.\n"
    )
    return """---
license: other
pretty_name: WildfireIA Anonymous Benchmark Release
task_categories:
- tabular-classification
- tabular-regression
- image-classification
- other
tags:
- wildfire
- benchmark
- geospatial
- multimodal
- croissant
size_categories:
- 10K<n<100K
---

# WildfireIA Anonymous Benchmark Release

This anonymous release contains the WildfireIA benchmark data used for review.
WildfireIA is an event-level benchmark for predicting whether a reported
Natural wildfire escapes initial attack from public information available at
fire discovery time.

## Contents

- `data/canonical/raw_feature_tables/`: canonical benchmark tables. These are
  the primary dataset artifact. They contain event-level tables, source-level
  feature tables, patch-level canonical tables, labels, splits, and manifests.
""" + cache_bullet + """\
- `code/`: anonymous copies of the cache generation, training, and summary
  scripts.
- `metadata/`: release manifest and cache generation commands.
- `croissant.json`: Croissant metadata with Responsible AI fields.

## Tasks

Task 1 predicts initial attack failure. The sample unit is one FPA-FOD Natural
wildfire event. Events with final size at most 10 ha are labeled 0, events with
final size at least 50 ha are labeled 1, and intermediate-size events are
excluded from the Task 1 supervised split.

Task 2 predicts remaining time-to-containment as a regression target,
`log(1 + containment_hours)`, using the same discovery-time input contract.

## Rebuilding Model-Ready Caches

The canonical tables can regenerate all official model-ready caches:

```bash
python code/dataloader.py \\
  --base_dir . \\
  --canonical_dir data/canonical/raw_feature_tables \\
  --output_dir data/model_ready \\
  --task ia_failure \\
  --representation all \\
  --weather_days 5 \\
  --input_protocol all \\
  --overwrite
```

Additional ablation caches are generated by changing `--input_protocol` and
`--weather_days`; see `metadata/cache_generation_commands.md`.

## Responsible Use

The benchmark is intended for reproducible scientific comparison and ablation
analysis. It should not be used as a standalone operational dispatch system
without agency validation. The data are public-source derived, but they include
wildfire locations, fire-station locations, roads, population density, and other
geospatial context.
"""


def build_data_licenses() -> str:
    return """# Data Licenses and Provenance

This release is a derived benchmark artifact built from public data sources.
Source-specific terms still apply.

- FPA-FOD wildfire occurrence data: USDA Forest Service Research Data Archive.
- FIRMS/VIIRS active fire detections: NASA FIRMS.
- gridMET meteorology and fire-danger variables.
- LANDFIRE vegetation, fuel, and topography rasters.
- OpenStreetMap roads and fire-station features. OSM-derived access features
  are subject to OpenStreetMap/ODbL terms.
- WorldPop population rasters.

The code in `code/` may be redistributed under the repository license chosen
for the anonymous submission. The data are distributed under source-compatible
terms rather than a single newly asserted license.
"""


def build_cache_commands() -> str:
    return """# Cache Generation Commands

All commands assume the current working directory is the dataset repository.

## Task 1 Full Input

```bash
python code/dataloader.py --base_dir . --canonical_dir data/canonical/raw_feature_tables --output_dir data/model_ready --task ia_failure --representation all --weather_days 5 --input_protocol all --overwrite
```

## Task 2 Full Input

```bash
python code/dataloader.py --base_dir . --canonical_dir data/canonical/raw_feature_tables --output_dir data/model_ready --task containment_time --representation all --weather_days 5 --input_protocol all --overwrite
```

## Task 1 Weather-History Ablation

```bash
for d in 1 2 3 4 5; do
  python code/dataloader.py --base_dir . --canonical_dir data/canonical/raw_feature_tables --output_dir data/model_ready --task ia_failure --representation all --weather_days ${d} --input_protocol weather --overwrite
done
```

## Task 1 FPA-FOD Metadata Plus One Source

```bash
for p in metadata_vegetation metadata_fuel metadata_topography metadata_access metadata_human; do
  python code/dataloader.py --base_dir . --canonical_dir data/canonical/raw_feature_tables --output_dir data/model_ready --task ia_failure --representation all --weather_days 5 --input_protocol ${p} --overwrite
done
```

## Task 1 Leave-One-Source-Out

```bash
for p in all_without_fire all_without_weather all_without_vegetation all_without_fuel all_without_topography all_without_access all_without_human; do
  python code/dataloader.py --base_dir . --canonical_dir data/canonical/raw_feature_tables --output_dir data/model_ready --task ia_failure --representation all --weather_days 5 --input_protocol ${p} --overwrite
done
```
"""


def build_upload_instructions() -> str:
    return """# Upload Instructions

This folder is a staging copy for the anonymous Hugging Face dataset repository.

Recommended upload workflow:

```bash
cd <local-project-root>
git xet install
git clone <anonymous-huggingface-dataset-ssh-url> <hf-clone>
rsync -a --info=progress2 <staging-folder>/ <hf-clone>/
cd <hf-clone>
git status --short
git add .
git commit -m "Anonymous WildfireIA dataset release"
git push
```

Before pushing, run a local identity scan for personal names, local usernames,
absolute machine paths, affiliations, and email addresses. The release should
only contain anonymous creator metadata during review.
"""


def build_croissant(release_root: Path, inventory: list[dict], include_task1_full_cache: bool) -> dict:
    canonical_dist = []
    for name in CANONICAL_FILES:
        path = release_root / "data/canonical/raw_feature_tables" / name
        is_parquet = name.endswith(".parquet")
        entry = {
            "@type": "cr:FileSet" if path.is_dir() else "cr:FileObject",
            "@id": f"canonical_{name.replace('/', '_')}",
            "name": name,
            "encodingFormat": "application/x-parquet" if is_parquet else "application/json",
            "contentUrl": f"{HF_RESOLVE}/data/canonical/raw_feature_tables/{name}",
            "contentSize": tree_size(path) if path.exists() else None,
        }
        if path.is_dir():
            entry["includes"] = "**/*.parquet" if is_parquet else "**/*"
        canonical_dist.append(entry)

    cache_dist = []
    if include_task1_full_cache:
        for cache in TASK1_FULL_CACHE_DIRS:
            path = release_root / "data/model_ready" / cache
            cache_dist.append(
                {
                    "@type": "cr:FileSet",
                    "@id": f"cache_{cache.replace('/', '_')}",
                    "name": cache,
                    "encodingFormat": "application/octet-stream",
                    "contentUrl": f"{HF_RESOLVE}/data/model_ready/{cache}",
                    "contentSize": tree_size(path) if path.exists() else None,
                    "description": "Task 1 full-input model-ready cache with NumPy arrays, sample indices, and metadata.",
                }
            )

    event_source = {"fileObject": {"@id": "canonical_fire_events_natural_2016_2020.parquet"}}
    master_source = {"fileObject": {"@id": "canonical_master_features_natural_2016_2020.parquet"}}
    patch_static_source = {"fileSet": {"@id": "canonical_event_static_patch_375m_natural_2016_2020.parquet"}}
    patch_weather_source = {"fileSet": {"@id": "canonical_event_weather_daily_patch_375m_natural_2016_2020.parquet"}}

    def field(record_set: str, name: str, data_type: str, source: dict, description: str = "") -> dict:
        item = {
            "@type": "cr:Field",
            "@id": f"{record_set}/{name}",
            "name": name,
            "dataType": data_type,
            "source": {**source, "extract": {"column": name}},
        }
        if description:
            item["description"] = description
        return item

    record_sets = [
        {
            "@type": "cr:RecordSet",
            "@id": "canonical_event_tables",
            "name": "canonical_event_tables",
            "description": "FPA-FOD event-level tables, labels, splits, and source-level features.",
            "dataType": "cr:Parquet",
            "key": [{"@id": "canonical_event_tables/fire_id"}],
            "field": [
                field("canonical_event_tables", "fire_id", "sc:Text", event_source, "FPA-FOD event identifier."),
                field("canonical_event_tables", "year", "sc:Integer", event_source, "Fire discovery year."),
                field("canonical_event_tables", "split", "sc:Text", event_source, "Chronological benchmark split."),
                field("canonical_event_tables", "ia_failure_label", "sc:Integer", event_source, "Task 1 binary label."),
                field("canonical_event_tables", "log_containment_hours", "sc:Float", event_source, "Task 2 regression target."),
                field("canonical_event_tables", "has_viirs_detection_1km_D", "sc:Integer", master_source, "Discovery-day VIIRS match indicator."),
            ],
        },
        {
            "@type": "cr:RecordSet",
            "@id": "canonical_patch_tables",
            "name": "canonical_patch_tables",
            "description": "Event-centered 375 m patch canonical tables for spatial and spatiotemporal representations.",
            "dataType": "cr:Parquet",
            "key": [
                {"@id": "canonical_patch_tables/fire_id"},
                {"@id": "canonical_patch_tables/cell_id"},
            ],
            "field": [
                field("canonical_patch_tables", "fire_id", "sc:Text", patch_static_source, "FPA-FOD event identifier."),
                field("canonical_patch_tables", "year", "sc:Integer", patch_static_source, "Fire discovery year."),
                field("canonical_patch_tables", "split", "sc:Text", patch_static_source, "Chronological benchmark split."),
                field("canonical_patch_tables", "cell_id", "sc:Text", patch_static_source, "Patch cell identifier."),
                field("canonical_patch_tables", "fbfm40", "sc:Integer", patch_static_source, "LANDFIRE fire behavior fuel model."),
                field("canonical_patch_tables", "evc", "sc:Float", patch_static_source, "LANDFIRE existing vegetation cover."),
                field("canonical_patch_tables", "elev", "sc:Float", patch_static_source, "Elevation."),
                field("canonical_patch_tables", "pop_density", "sc:Float", patch_static_source, "WorldPop population density."),
                field("canonical_patch_tables", "relative_day", "sc:Integer", patch_weather_source, "Weather day relative to discovery day."),
                field("canonical_patch_tables", "tmmx", "sc:Float", patch_weather_source, "Daily maximum temperature."),
            ],
        },
    ]
    if include_task1_full_cache:
        record_sets.append(
            {
                "@type": "cr:RecordSet",
                "@id": "task1_full_model_ready_caches",
                "name": "task1_full_model_ready_caches",
                "description": "Derived full-input caches for Task 1 initial attack failure prediction.",
                "dataType": "cr:FileSet",
                "field": [
                    {
                        "@type": "cr:Field",
                        "@id": "task1_full_model_ready_caches/cache_path",
                        "name": "cache_path",
                        "dataType": "sc:Text",
                        "value": "data/model_ready/ia_failure",
                    }
                ],
            }
        )

    return {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "sc": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "rai": "http://mlcommons.org/croissant/RAI/",
            "prov": "http://www.w3.org/ns/prov#",
        },
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "name": "WildfireIA Anonymous Benchmark Release",
        "alternateName": "WildfireIA",
        "description": (
            "Event-level multimodal benchmark for predicting whether a Natural wildfire "
            "escapes initial attack from public discovery-time data."
        ),
        "url": f"https://huggingface.co/datasets/{DATASET_REPO_ID}",
        "license": f"{HF_RESOLVE}/DATA_LICENSES.md",
        "datePublished": date.today().isoformat(),
        "creator": [{"@type": "Organization", "name": "Anonymous Authors"}],
        "keywords": ["wildfire", "initial attack", "benchmark", "geospatial", "multimodal"],
        "distribution": canonical_dist + cache_dist,
        "recordSet": record_sets,
        "rai:dataLimitations": (
            "The benchmark covers 2016--2020 contiguous United States Natural wildfire events "
            "that pass quality filters. It does not represent Alaska, Hawaii, non-natural fires, "
            "or non-US initial attack systems. The binary failure label is a final-size proxy."
        ),
        "rai:dataBiases": (
            "Potential biases include FPA-FOD reporting practices, VIIRS overpass and cloud/smoke "
            "limitations, gridMET gridding uncertainty, LANDFIRE static-layer mismatch, OSM mapping "
            "completeness, and WorldPop population-model uncertainty."
        ),
        "rai:personalSensitiveInformation": (
            "No direct personal identifiers are included. The data include public wildfire locations, "
            "dates, roads, fire-station locations, and population density, which are geospatially "
            "sensitive contextual information."
        ),
        "rai:dataUseCases": (
            "Reproducible benchmark evaluation, representation comparison, source ablation, and "
            "scientific analysis of public discovery-time signals for wildfire initial attack."
        ),
        "rai:dataSocialImpact": (
            "The benchmark may support transparent research on early wildfire risk ranking. Misuse as "
            "an unvalidated operational dispatch tool could reinforce geographic or reporting biases."
        ),
        "rai:hasSyntheticData": False,
        "prov:wasDerivedFrom": [
            "FPA-FOD",
            "NASA FIRMS/VIIRS",
            "gridMET",
            "LANDFIRE",
            "OpenStreetMap",
            "WorldPop",
        ],
        "prov:wasGeneratedBy": (
            "Canonicalization with pipeline.py followed by deterministic cache generation with "
            "dataloader.py using chronological splits and discovery-time input restrictions."
        ),
        "wildfireia:fileInventory": inventory,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=Path("."))
    parser.add_argument("--release_dir", type=Path, default=Path("./release/Anonymous-WildfireIA"))
    parser.add_argument("--copy", action="store_true", help="Copy files instead of hardlinking when possible.")
    parser.add_argument("--include_task1_full_cache", action="store_true", help="Include the large Task 1 full-input model-ready caches.")
    args = parser.parse_args()

    base = args.base_dir.resolve()
    release = args.release_dir.resolve()
    hardlink = not args.copy
    release.mkdir(parents=True, exist_ok=True)

    canonical_src = base / "data/cache/raw_feature_tables"
    canonical_dst = release / "data/canonical/raw_feature_tables"
    for name in CANONICAL_FILES:
        src = canonical_src / name
        if not src.exists():
            raise FileNotFoundError(src)
        copy_or_link_tree(src, canonical_dst / name, hardlink=hardlink)

    model_src = base / "data/cache/model_ready"
    model_dst = release / "data/model_ready"
    if args.include_task1_full_cache:
        for cache in TASK1_FULL_CACHE_DIRS:
            src = model_src / cache
            if not src.exists():
                raise FileNotFoundError(src)
            copy_or_link_tree(src, model_dst / cache, hardlink=hardlink)
    elif model_dst.exists():
        shutil.rmtree(model_dst)

    code_dir = release / "code"
    for name in CODE_FILES:
        src = base / name
        text = sanitized_code(src.read_text(encoding="utf-8"))
        write_text(code_dir / name, text)

    write_text(release / "README.md", build_readme(args.include_task1_full_cache))
    write_text(release / "DATA_LICENSES.md", build_data_licenses())
    write_text(release / "UPLOAD_INSTRUCTIONS.md", build_upload_instructions())
    write_text(release / "metadata/cache_generation_commands.md", build_cache_commands())
    write_text(
        release / ".gitattributes",
        "\n".join(
            [
                "*.npy filter=lfs diff=lfs merge=lfs -text",
                "*.npz filter=lfs diff=lfs merge=lfs -text",
                "*.parquet filter=lfs diff=lfs merge=lfs -text",
                "*.pt filter=lfs diff=lfs merge=lfs -text",
                "",
            ]
        ),
    )

    inventory = []
    for path in sorted((release / "data").rglob("*")):
        if path.is_file():
            inventory.append(
                {
                    "path": rel(path, release),
                    "bytes": path.stat().st_size,
                    "hardlinked": path.stat().st_nlink > 1,
                }
            )

    parquet_tables = {}
    for name in CANONICAL_FILES:
        path = canonical_dst / name
        if name.endswith(".parquet"):
            parquet_tables[name] = parquet_summary(path, release)

    task1_cache_total = (
        sum(tree_size(model_dst / cache) for cache in TASK1_FULL_CACHE_DIRS)
        if args.include_task1_full_cache
        else 0
    )
    manifest = {
        "dataset": "WildfireIA anonymous release",
        "created_date": date.today().isoformat(),
        "release_dir": ".",
        "canonical_root": "data/canonical/raw_feature_tables",
        "task1_full_cache_root": "data/model_ready/ia_failure" if args.include_task1_full_cache else None,
        "canonical_total_bytes": tree_size(canonical_dst),
        "task1_full_cache_total_bytes": task1_cache_total,
        "parquet_tables": parquet_tables,
        "task1_full_caches": TASK1_FULL_CACHE_DIRS if args.include_task1_full_cache else [],
        "file_count": len(inventory),
    }
    write_text(release / "metadata/release_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_text(release / "metadata/file_inventory.json", json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    write_text(
        release / "croissant.json",
        json.dumps(build_croissant(release, inventory, args.include_task1_full_cache), indent=2, sort_keys=True) + "\n",
    )

    write_text(
        release / "metadata/identity_check_summary.md",
        "# Identity Check Summary\n\nNo author names, affiliations, emails, or local absolute paths are intended in this release. Run the repository-level scan before uploading.\n",
    )

    print(f"Prepared release at: {release}")
    print(f"Canonical bytes: {manifest['canonical_total_bytes']:,}")
    print(f"Task1 full cache bytes: {manifest['task1_full_cache_total_bytes']:,}")
    print(f"Files under data/: {manifest['file_count']:,}")


if __name__ == "__main__":
    main()
