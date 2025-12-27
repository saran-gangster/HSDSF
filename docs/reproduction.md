# Reproduction (Local Data + Colab Training)

This repo is set up so **local runs only generate data** (CPU friendly), while **training happens in Google Colab (T4)**.

## Local: generate the dataset artifacts

From the repo root:

- Generate runs + splits + windowed NPZ (no training):
  - `bash experiments/run_data_only.sh`

Outputs:
- Runs: `data/fusionbench_sim/runs/run_*/{telemetry.csv,meta.json,intervals.csv}`
- Split manifests: `data/fusionbench_sim/splits/*.json`
- Windowed tensors: `data/fusionbench_sim/processed/<split_name>/windows_{train,val,test}.npz`

## Colab: train models (no code runs here locally)

1) Upload the whole repo (or at least these folders) to Colab runtime / Drive:
- `data/fusionbench_sim/processed/`
- `data/fusionbench_sim/runs/` (needed for `intervals.csv` during evaluation)
- `dynamic/`, `fusion/`, `evaluation/` (training + evaluation scripts)

2) Install requirements in Colab:
- `pip install -r requirements-colab.txt`

3) Train dynamic expert ensemble (example commands)

Choose one split name, e.g. `unseen_workload`:

- `python dynamic/train_dynamic.py --processed-dir data/fusionbench_sim/processed/unseen_workload --out-dir artifacts/dynamic/unseen_workload`

4) (Later) Train static expert / fusion gate

Static is optional for simulator-only binaries. If you have real binaries and populate `data/fusionbench_sim/binaries/binaries.csv` with paths, you can run:

- `python static/extract_static.py --binaries-csv data/fusionbench_sim/binaries/binaries.csv --out data/fusionbench_sim/binaries/static_features.csv`
- `python static/train_static.py --features data/fusionbench_sim/binaries/static_features.csv --out-dir artifacts/static`

Then fusion:

- `python fusion/train_fusion.py --processed-dir data/fusionbench_sim/processed/unseen_workload --dynamic-dir artifacts/dynamic/unseen_workload --static-dir artifacts/static --out-dir artifacts/fusion/unseen_workload`

## Notes

- Split protocol is **run-level** by design. Do not randomly shuffle windows across runs.
- Mask features (`mask_*`) are kept **binary** (not standardized), so they can be used as gate meta-context.
