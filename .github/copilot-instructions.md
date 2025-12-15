# Copilot instructions (HSDSF)

## Big picture (what this repo is)
- This repo is **plan-first**, but Phase 2 now has runnable code: start with [README.md](README.md) for the overall phased PoC, then use [phase2/README.md](phase2/README.md) as the Phase 2 runbook.
- Target PoC: a **hybrid static + dynamic security framework** for supply-chain anomaly detection on NVIDIA Jetson AGX Xavier.
  - **Phase 2 (implemented):** off-device static analysis of binaries (binutils extraction → chunking → LLM scoring → report).

## Repo structure (current)
- [README.md](README.md): overall architecture + phased deliverables (static LLM analysis → telemetry → anomaly model → TensorRT deployment).
- [phase2/README.md](phase2/README.md): how to build sample binaries + run extraction / full analysis.
- [phase2/planforphase2.md](phase2/planforphase2.md): background Phase 2 design notes (chunking + prompt structure).
- `phase2/analyze_binary.py`: the end-to-end Phase 2 runner.

## Where to put Phase 2 code
- Keep all Phase 2 implementation and outputs under `phase2/`.
- Don’t commit generated artifacts: `phase2/tmp/` and `phase2/reports/` are git-ignored (see `phase2/.gitignore`).

## Phase 2 workflow (what agents should follow)
- System prerequisites: Unix-like binutils (`objdump`, `readelf`, `nm`, `strings`, `file`) + a C compiler for sample binaries.
  - On Windows, prefer WSL so `objdump/readelf/nm/strings` behave as expected.
- Python deps: install from `phase2/requirements.txt`.
- Run extraction only (no API): `python phase2/analyze_binary.py <binary> --skip-llm`
- Run full analysis: requires `CEREBRAS_API_KEY` (loaded via `python-dotenv` from a repo-root `.env`).
  - Optional sanity check: `python phase2/test_api.py`
- Outputs are written to `phase2/reports/<binary_name>_<sha8>/`:
  - `extraction_summary.json`, `function_reports.json`, `binary_report.md`

## Phase 2 conventions (implemented in code)
- Extraction mirrors the plan and is implemented in `extract_artifacts()` in `phase2/analyze_binary.py` (writes `.dis/.strings/.meta/.symbols/.file` under `phase2/tmp/`).
- Chunking is by function headers in `objdump` output; if the binary is stripped and few functions are found, it falls back to section/region chunking.
- Default chunk size is tuned for LLM context (`--max-lines-per-chunk` default 350) and analysis is limited by `--max-functions`.
- Risk triage is controlled by `--risk-threshold` (default 0.4) and `--batch-size`.

## External integrations / configuration
- LLM provider used by the current Phase 2 implementation: **Cerebras** (`cerebras-cloud-sdk`, default model `zai-glm-4.6`).
- Read API key from env var `CEREBRAS_API_KEY` (prefer a local `.env`; never hardcode secrets in code or docs).

## Dev workflow assumptions (important)
- Prefer runnable scripts over frameworks: Phase 2 is a CLI runner (`phase2/analyze_binary.py`) with artifact + report directories.
- Validation baseline lives in Phase 2 assets:
  - `phase2/firmware_clean.c`, `phase2/firmware_trojan.c`, plus the expected behavior in `phase2/ground_truth.md`.

## When adding new code
- Keep changes tightly scoped to what’s described in [README.md](README.md) and the Phase 2 docs in `phase2/`.
- Don’t introduce additional phases, UIs, or services unless a doc in this repo explicitly calls for it.
