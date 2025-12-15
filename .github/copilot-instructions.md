# Copilot instructions (HSDSF)

## Big picture (what this repo is)
- This repo is currently **plan-first**: the primary “source of truth” is the project plan in [README.md](README.md) and the Phase 2 execution notes in [phase2/planforphase2.md](phase2/planforphase2.md).
- Target PoC: a **hybrid static + dynamic security framework** for supply-chain anomaly detection on NVIDIA Jetson AGX Xavier.
  - **Phase 2 (current focus):** off-device static analysis of binaries via disassembly/strings/metadata + LLM-assisted reporting.

## Repo structure (current)
- [README.md](README.md): overall architecture + phased deliverables (static LLM analysis → telemetry → anomaly model → TensorRT deployment).
- [phase2/planforphase2.md](phase2/planforphase2.md): concrete Phase 2 pipeline spec (tools, chunking, prompts, expected scripts).

## Where to put Phase 2 code
- Keep all Phase 2 implementation scripts and outputs under `phase2/` (this repo’s Phase 2 “home”).
- Preferred entrypoints:
  - `phase2/analyze_binary.py` (end-to-end runner)
  - `phase2/extract_binary_info.py` (optional helper; may be merged into `analyze_binary.py`)

## Phase 2 pipeline (implement to match the plan)
- Data flow:
  1) Input binary (prefer ELF; PE is optional per plan)
  2) Extract artifacts using binutils tooling (`objdump`, `readelf`, `nm`, `strings`)
  3) Chunk disassembly by function/region (function headers like `<function_name>:` in `objdump` output)
  4) Send each chunk to the LLM with a structured prompt
  5) Aggregate results into a **structured report** with evidence locations (addresses/functions/strings) + confidence/rationale

## Conventions to follow (from docs)
- Prefer script names referenced in Phase 2 doc when creating implementation (place them under `phase2/`; see above).
- Extraction commands and outputs should mirror the plan:
  - `objdump -d -Mintel -w <bin> > <bin>.dis`
  - `strings <bin> > <bin>.strings` (optionally `strings -t x` if you need addresses)
  - `readelf -h -S -s <bin> > <bin>.meta`
  - `nm -C <bin> > <bin>.symbols`
- Chunking targets (initial heuristic per plan): ~200–400 lines of disassembly per chunk; split large functions into `name#1`, `name#2`, …
- Use a typed data model for chunks (Phase 2 doc shows a `FunctionChunk` Pydantic sketch with `name`, `start_addr`, `disassembly`, `strings_nearby`).

## External integrations / configuration
- LLM provider in Phase 2 notes: **Gemini**.
  - Read the API key from env var `GEMINI_API_KEY`.
  - Standardize on the `google-genai` SDK only.
    - Install: `pip install google-genai`
    - Imports used in the plan:
      - `from google import genai`
      - `from google.genai import types`

## Dev workflow assumptions (important)
- The docs assume **Unix-like tooling** (binutils, gcc, etc.). If developing on Windows, prefer **WSL** or a Linux environment so `objdump/readelf/nm/strings` behave as expected.
- The repo does not currently define a build/test runner; keep early implementations runnable as simple scripts (e.g., `python analyze_binary.py <path-to-binary>`), and align outputs with the report formats described in the docs.

## When adding new code
- Keep changes tightly scoped to what’s described in [README.md](README.md) and [phase2/planforphase2.md](phase2/planforphase2.md).
- Don’t introduce additional phases, UIs, or services unless a doc in this repo explicitly calls for it.
