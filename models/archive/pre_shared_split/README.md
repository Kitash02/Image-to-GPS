# Archive: Pre-Shared-Split Historical Runs

This folder preserves three early backbone runs from the initial pipeline (before the
shared split manifest and deterministic `--seed` / `--shared_split_path` feature was
introduced in `src/train.py`). They are kept as **historical evidence** and are **not**
part of the current benchmark reported in `data/report_run_summary.csv`. The current
headline numbers are produced by the timestamped folders
`models/resnet50_20260305_2009/`, `models/efficientnet_b0_20260306_2115/`,
`models/convnext_tiny_20260307_1319/`, and the fine-tuned
`models/finetune_photos_runs/resnet50_20260315_171307/`.

## What is here

- `resnet50_early/` — earlier ResNet50 training run.
  - `config.json` preserves the *early* hyperparameters that included a
    `gps_weight=0.8`, `area_weight=0.2` composite-loss experiment that was later
    dropped in favour of a pure Huber-on-coordinates loss (see "What Did Not Work"
    in the report).
  - `history.csv`, `test_results.csv`, `failure_summary.txt`.
- `efficientnet_b0_early/` — `history.csv`, `test_results.csv`, `failure_summary.txt`.
- `convnext_tiny_early/` — `history.csv`, `test_results.csv`, `failure_summary.txt`.

## Why the `failure_summary.txt` files matter

Each file contains a per-device breakdown on the early test split:

```
iPhone mean error: nan m
iPhone median error: nan m
iPhone acc@5m: nan%

Android mean error: <value> m
Android median error: <value> m
Android acc@5m: <value>%
```

The `nan` on iPhone indicates there were no iPhone images in that early split
(the subset was dominated by Android captures, partly due to HEIC handling at
that stage). This is the raw evidence used to discuss **phone robustness** in
the report's Limitations / Future Work section. It is intentionally preserved
here because the current benchmark files do not break results down by device.

## Why these runs are not the current benchmark

These folders predate two changes:

1. `src/train.py` now builds a deterministic split from `--seed` and persists it
   via `--shared_split_path`, so all backbones evaluate on the same test images.
2. `run_experiments.py` now forwards the seed and shared split path to
   `src/train.py`, so a rerun is reproducible cross-backbone.

Using the early runs as the official benchmark would mean each backbone was
scored on a different (implicitly seeded) test split, so the pairwise
backbone comparison would not be strictly fair.

## Don't delete

These files are small and add historical context that the report narrative
refers to. Do not delete them; delete only if an instructor requests a smaller
repo.
