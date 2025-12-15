# Phase 2 (Static LLM Binary Analysis)

## Quick start

1) Install system tools (Linux): `binutils`, `file`, `xxd`, and a C compiler.

2) Create a Python venv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r phase2/requirements.txt
```

3) Set up Cerebras API key in `.env` (already configured):

```bash
# .env file contains:
CEREBRAS_API_KEY=csk-wfp4yhk8xyhc35c2rvwdrx2yhy4rkk49rtdx83m3jkkhj2rm
```

4) Build the sample binaries:

```bash
gcc -static -O2 -s phase2/firmware_clean.c  -o phase2/firmware_clean
gcc -static -O2 -s phase2/firmware_trojan.c -o phase2/firmware_trojan
```

5) Run extraction-only (no API calls):

```bash
python phase2/analyze_binary.py phase2/firmware_clean --skip-llm
python phase2/analyze_binary.py phase2/firmware_trojan --skip-llm
```

6) Run full analysis with Cerebras zai-glm-4.6:

```bash
python phase2/analyze_binary.py phase2/firmware_trojan --max-functions 20 --batch-size 8
python phase2/analyze_binary.py phase2/firmware_clean --max-functions 20 --batch-size 8
```

## Results

Reports are generated in `phase2/reports/<binary_name>_<sha>/`:
- `extraction_summary.json` - metadata about the binary and extraction
- `function_reports.json` - per-function LLM analysis with risk scores
- `binary_report.md` - executive summary with top findings

## Validation

The trojan binary shows clear differentiation:
- **Trojan**: avg risk 0.69, max 0.90, 8 high-risk functions (>0.7)
- **Clean**: avg risk 0.17, max 0.60, 0 high-risk functions

Ground truth backdoors in `firmware_trojan`:
- Hardcoded credentials: `debug` / `letmein!`
- Environment variable bypass: `FW_DEBUG_KEY=0xDEADBEEF`
