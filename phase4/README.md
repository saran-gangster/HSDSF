# Phase 4: Anomaly Model Training

This folder hosts the training pipeline for the time-series anomaly detector built from Phase 3 telemetry.

## What this delivers
- Sliding-window preprocessing over Phase 3 runs
- Unsupervised LSTM autoencoder trained on benign classes (idle + normal by default)
- Reconstruction-error thresholding for anomaly scoring (trojans expected to exceed threshold)
- Artifacts: `model.pt`, `scaler.pkl`, `features.json`, `threshold.json`, optional `model.onnx`

## Quick start (local or Colab GPU)
1) Install deps (CPU is fine; GPU recommended for speed):
```bash
pip install -r phase4/requirements.txt
```

2) Train with defaults (5s windows, 0.5s stride, idle+normal as benign):
```bash
python phase4/train_anomaly.py \
  --runs-dir phase3/data/runs \
  --output-dir phase4/artifacts \
  --device cuda \
  --export-onnx
```

Key flags:
- `--window-size` (samples): default 50 → 5s at 10 Hz
- `--window-stride`: default 5 → 0.5s step
- `--train-labels`: benign classes (default `idle normal`)
- `--threshold-percentile`: percentile of train reconstruction error used as decision threshold (default 99)

3) Outputs land in `phase4/artifacts/`:
- `model.pt`: LSTM autoencoder weights
- `model.onnx`: export for TensorRT conversion (when `--export-onnx` is set)
- `scaler.pkl`: fitted StandardScaler on benign windows
- `features.json`: ordered feature list
- `threshold.json`: anomaly threshold derived from train errors
- `metrics.json`: training/eval summary (losses, AUC if labels available)

## Running in Google Colab (GPU)
1) Start a GPU runtime (Runtime → Change runtime type → GPU).
2) Pull the repo (or upload a zip) and install requirements:
```bash
!pip install -r phase4/requirements.txt
```
3) Ensure Phase 3 data is present at `phase3/data/runs/` (upload or `git lfs` if needed).
4) Run the training command above with `--device cuda --export-onnx`.
5) Download `phase4/artifacts/` for TensorRT conversion on Jetson.

## Next steps toward deployment
- Convert `model.onnx` to a TensorRT engine on Jetson (e.g., `trtexec --onnx=model.onnx --saveEngine=model.plan`).
- Implement real-time monitor that mirrors Phase 3 feature extraction, applies `scaler.pkl`, and scores windows against `threshold.json` (see Phase 4 tasks in repo root README).
