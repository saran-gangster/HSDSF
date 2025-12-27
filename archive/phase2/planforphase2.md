Plan to execute Phase 2 now (no Jetson, using Gemini Studio)
-------------------------------------------------------------

We can do all of Phase 2 entirely on laptop; nothing in that phase actually depends on Jetson hardware. You just need binaries, standard Unix tooling, and the Gemini API.

### 1.1. Concrete goals for Phase 2 (adapted)

While waiting for hardware, make Phase 2 slightly more general:

- Input: any ELF or PE binary (but ideally ELF, to match Jetson later).
- Output: structured LLM report with:
  - Suspicious patterns / potential backdoors.
  - Evidence locations (addresses, function names, strings).
  - Confidence scores and rationales.
- Demonstration:
  - At least one “clean” synthetic firmware-like binary.
  - At least one “Trojan” version with known, documented backdoor.
  - Show that the Trojan gets flagged more strongly than the clean version.

### 1.2. Step 1 – Environment setup (laptop + Gemini)

**Tools to install**

- System tools:
  - `binutils`: `objdump`, `readelf`, `nm`, `strings`
  - `file`, `xxd`
- (Optional, but helpful later) One of:
  - `ghidra`, `rizin/radare2`, or `binaryninja` (for richer IR/decompilation)
- Python env (e.g., `venv`):
  - `python -m venv venv && source venv/bin/activate`
  - `pip install google-generativeai rich pydantic orjson`

**Configure Gemini**

- In Gemini AI Studio, create an API key.
- In Python:

```python
# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-3-pro-preview"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""INSERT_INPUT_HERE"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinkingConfig: {
            thinkingLevel: "HIGH",
        },
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()

```

You’ll use this model in the Phase 2 pipeline.

### 1.3. Step 2 – Collect and prepare binaries (no hardware needed)

You don’t need Jetson to get firmware-like binaries. Use:

1. **Realistic “firmware-style” binaries** (read-only analysis)
   - Compile small static ELF programs that look like firmware:
     - Command dispatcher
     - Network listener
     - Config file parser
   - Or grab existing small daemons from your system:
     - `/usr/sbin/sshd`, `/usr/sbin/cron`, `/bin/busybox` if present.
   - Optional: cross-compile for AArch64 to match Jetson’s architecture:
     - `aarch64-linux-gnu-gcc -static -O2 -s main.c -o firmware_clean`

2. **Synthetic clean vs Trojan binaries** (you control ground truth)

Create one base C program and one with a backdoor:

- `firmware_clean.c` (sketch):

```c
#include <stdio.h>
#include <string.h>

int authenticate(const char *user, const char *pass) {
    if (strcmp(user, "admin") == 0 && strcmp(pass, "admin123") == 0)
        return 1;
    return 0;
}

void handle_command(const char *cmd) {
    if (strcmp(cmd, "status") == 0) {
        printf("OK\n");
    } else if (strcmp(cmd, "reboot") == 0) {
        printf("Rebooting...\n");
    } else {
        printf("Unknown command\n");
    }
}

int main() {
    char user[32], pass[32], cmd[64];
    printf("Username: ");
    scanf("%31s", user);
    printf("Password: ");
    scanf("%31s", pass);
    if (!authenticate(user, pass)) {
        printf("Auth failed\n");
        return 1;
    }
    while (1) {
        printf("cmd> ");
        if (scanf("%63s", cmd) != 1) break;
        handle_command(cmd);
    }
}
```

- `firmware_trojan.c`: add a hidden backdoor:

```c
int authenticate(const char *user, const char *pass) {
    // Legit path
    if (strcmp(user, "admin") == 0 && strcmp(pass, "admin123") == 0)
        return 1;

    // BACKDOOR 1: magic username / password
    if (strcmp(user, "debug") == 0 && strcmp(pass, "letmein!") == 0)
        return 1;

    // BACKDOOR 2 (optional): magic environment variable
    const char *key = getenv("FW_DEBUG_KEY");
    if (key && strcmp(key, "0xDEADBEEF") == 0)
        return 1;

    return 0;
}
```

- Compile both:
  - `gcc -static -O2 -s firmware_clean.c   -o firmware_clean`
  - `gcc -static -O2 -s firmware_trojan.c -o firmware_trojan`

Document the exact backdoor details in a `ground_truth.md` file:
- File, function, source lines, and intended secret logic.

### 1.4. Step 3 – Extraction pipeline (disassembly, strings, metadata)

Implement a small Python module `extract_binary_info.py` (or just inside `analyze_binary.py`) that:

1. **Runs tools**:

```bash
objdump -d -Mintel -w firmware_trojan   > firmware_trojan.dis
strings firmware_trojan                 > firmware_trojan.strings
readelf -h -S -s firmware_trojan       > firmware_trojan.meta
nm -C firmware_trojan                  > firmware_trojan.symbols
```

2. **Organizes outputs**:
   - `disassembly`: raw `objdump -d` text.
   - `strings`: one per line.
   - `metadata`: headers/sections/symbols.

3. **Optionally**:
   - Use `radare2` or `ghidra` headless to get:
     - Per-function disassembly.
     - Call graph or function list.
   - But you can start with pure `objdump`.

This step is completely hardware agnostic.

### 1.5. Step 4 – Design chunking and data model

Even with long context, you want systematic, reusable chunking.

**Core idea:** “Per-function” or “per-region” chunks with a fixed prompt template.

1. **Identify functions / regions**
   - Parse the `objdump` output:
     - Lines like: `<function_name>:` after an address mark function starts.
   - Build a structure:

```python
class FunctionChunk(BaseModel):
    name: str
    start_addr: str
    disassembly: str
    strings_nearby: List[str]
```

2. **Associate relevant strings**
   - Simple heuristic: if a string’s address (from `strings -t x`) is within the function’s `.rodata` range, or just share all global strings for now.

3. **Chunk size policy**
   - Limit each chunk to, say, 200–400 lines of assembly.
   - If a function is huge, break it into multiple segments (`function_name#1`, `#2`, …).

4. **Global context snippet**
   - Prepare a short “binary-level” summary chunk:
     - File name, architecture, high-level metadata.
     - Top N interesting strings (e.g., containing `key`, `password`, `debug`, `admin`, etc.).
   - This is given to the LLM in the system or preamble section to orient it.

### 1.6. Step 5 – Prompt templates for Gemini

Define at least two prompt types:

#### 1. Function-level analysis prompt

Used for each `FunctionChunk`.

**System / prefix instruction:**

> You are a security analyst specialized in reverse engineering firmware binaries.  
> You are given:
> - Disassembled code for a single function.
> - Some relevant strings if available.
> Your goal is to identify suspicious behavior, potential backdoors, or security-relevant logic.  
> Focus on:
> - Authentication, authorization, and credential checks
> - Hidden commands or magic values
> - Hard-coded secrets, keys, or debug bypasses
> - Unusual branching on constant values or environment variables
> - Networking, file I/O, process control with unusual conditions  
> Return a structured JSON object as specified.

**User content template (Python f-string):**

```text
[BINARY CONTEXT]
Binary name: {binary_name}
Architecture: {arch}
Notable global strings: {notable_strings}

[FUNCTION DISASSEMBLY]
Function: {func.name}
Start address: {func.start_addr}

{func.disassembly}

[OPTIONAL STRINGS NEAR THIS FUNCTION]
{func.strings_nearby}

[INSTRUCTIONS]
1. Summarize what this function likely does.
2. Identify any clearly security-relevant operations.
3. Flag any behavior that could be a backdoor or Trojan trigger.
4. Highlight any magic constants, suspicious strings, or hidden branches.
5. Provide a numeric risk_score between 0 and 1 (0 = benign, 1 = clearly malicious).
6. Provide specific evidence: line snippets or addresses.

Return JSON with keys:
- "function_name": str
- "summary": str
- "security_relevant": bool
- "suspicious": bool
- "risk_score": float
- "findings": [ { "type": str, "description": str, "evidence": str } ]
- "notes": str
```

Call Gemini with `response_mime_type="application/json"` and a matching response schema if you want strict JSON.

#### 2. Binary-level aggregation prompt

After all function analyses, send a compact summary of findings to Gemini for a final verdict:

- Inputs:
  - Top K suspicious functions (e.g., `risk_score > 0.4`).
  - High-level metadata and strings.
- Ask Gemini to:
  - Describe overall behavior.
  - Identify whether there is strong evidence of backdoor.
  - Point out which functions and why.
  - Suggest follow-up manual reverse engineering steps.

### 1.7. Step 6 – Implementation plan for `analyze_binary.py`

Implement in phases:

1. **CLI skeleton**
   - `python analyze_binary.py firmware_clean --model gemini-3-pro-preview --out reports/firmware_clean.json`

2. **Extraction stage**
   - Run `objdump`, `strings`, `readelf`, etc.
   - Cache outputs in `tmp/` to avoid recomputing.

3. **Parsing and chunking**
   - Parse disassembly into `FunctionChunk` objects.
   - Heuristic: discard tiny functions (e.g., < 5 lines) or PLT stubs.

4. **LLM client**
   - A small class that wraps Gemini API calls, handles retries, and rate limiting.
   - Function: `analyze_function_chunk(chunk: FunctionChunk) -> FunctionReport`.

5. **Parallelization (optional, but helpful)**
   - Use `asyncio` or `concurrent.futures` to analyze multiple functions concurrently, respecting rate limits.

6. **Aggregation**
   - Combine `FunctionReport`s into:
     - A JSON file with per-function details.
     - A markdown summary report.
   - Invoke the binary-level aggregation prompt with summarized per-function data.

7. **Deterministic run**
   - Use a fixed `temperature` and `top_p` for reproducibility.
   - Persist raw LLM responses for debugging.

### 1.8. Step 7 – Evaluation without hardware

You can fully validate this phase now:

1. **Baseline runs**
   - Run `analyze_binary.py firmware_clean` and `firmware_trojan`.
   - Verify:
     - The backdoor function is flagged in `firmware_trojan`.
     - The analogous function in `firmware_clean` has significantly lower `risk_score`.

2. **Sensitivity tests**
   - Make variant Trojans:
     - Slightly obfuscated backdoor (e.g., compare hash of password instead of raw string).
     - Backdoor tied to environment variable only.
   - See how robust Gemini is to simple obfuscations.

3. **False positive checks**
   - Analyze a small library (`libc` fragment, or a simple network tool) to measure how many benign functions get high `risk_score`.

Save all this evidence (reports + your validation notes). It becomes part of your PoC doc and later comparison when you move to actual Jetson binaries.

---

2. Plan for an industry-grade, scalable LLM-based static analysis pipeline
--------------------------------------------------------------------------

Below is an architecture that could scale to “top-tier” usage (multi-team, CI integration, many binaries), followed by a concrete “distilled” subset you can realistically implement in this PoC.

### 2.1. Requirements for an industry-grade pipeline

- **Scalable**: Handle thousands of binaries and millions of functions; bounded cost.
- **Reliable**: Structured outputs, low hallucination, versioned models.
- **Auditable**: Every finding traceable to evidence (addresses, bytes, disassembly).
- **Composable**: Works with classic static tooling (IDA/Ghidra/Rizin, symbolic exec).
- **Integrable**: Fits into CI/CD and supply-chain workflows.

### 2.2. High-level architecture

Think of three main layers:

1. **Static Analysis Core (non-LLM)**
   - Binary ingestion
   - Normalization & triage
   - Disassembly & (optionally) decompilation
   - Basic heuristics & rule-based detectors

2. **LLM Orchestration Layer**
   - Chunking & context framing
   - Multi-stage LLM workflows (triage → deep analysis → cross-correlation)
   - Model routing (cheap vs expensive models, safety filters)
   - Response validation & post-processing

3. **Storage, Search, and UX**
   - Findings database (per-binary and per-function)
   - Vector search / embeddings store (for code patterns)
   - Web UI / dashboards
   - CI/CD hooks and APIs

### 2.3. Detailed pipeline stages

#### Stage 0 – Ingestion and normalization

- Accept input:
  - Raw firmware images (router, IoT, ECU)
  - Single binaries (ELF, PE)
- Steps:
  - Identify format and architecture with `file`, `readelf`, etc.
  - Unpack firmware (e.g., `binwalk`, `firmware-mod-kit`).
  - Deduplicate libraries using hash-based identity.
- Output:
  - A normalized “project” with a list of binaries, their paths, and architectures.

#### Stage 1 – Baseline static analysis (classic tools)

Use prior art first; LLMs work on their outputs:

- Disassembly and function discovery:
  - Ghidra/IDA/Rizin headless; emit:
    - Function list, symbols, control flow graphs (CFGs).
    - Cross-references (XREFs).
- Decompilation (if available):
  - Pseudo-C or IR (e.g., Ghidra’s p-code) per function.
- Metadata enrichment:
  - Imported libraries, syscalls used, section sizes, relocation info.
  - Strings with addresses and reference counts.

Store all of this in a structured form (e.g., PostgreSQL + object storage for text blobs).

#### Stage 2 – Heuristic pre-filtering and scoring

You must not send every function to an LLM; filter aggressively:

- Compute static “interest scores”:
  - Uses network APIs, crypto APIs, file or process control.
  - Contains suspicious strings: `"password"`, `"key"`, `"debug"`, `"root"`, etc.
  - Large amounts of constant data (possible embedded keys).
  - Control flow patterns: multiple nested branches, large switch tables, state machines.
- Rule-based detectors:
  - YARA rules for known malware families.
  - Regex-based checks in decompiled code (e.g., insecure APIs like `gets`, `strcpy`).
- Rank functions by interest score and select top X% for LLM analysis, plus a random sample.

#### Stage 3 – LLM analysis tier (multi-stage)

Use multiple LLM passes with increasing depth:

1. **Coarse-grain triage (cheap model or smaller context)**
   - Input:
     - Short excerpt of decompiled code + summary metadata.
   - Output:
     - `is_security_relevant`, `risk_score_coarse`, `tags` (e.g., `auth`, `network`, `crypto`).
   - Purpose:
     - Further reduce volume for deep analysis.

2. **Deep function review (full-size model like gemini-3-pro-preview)**
   - Input:
     - Full decompiled code for the function.
     - Surrounding cross-references, key strings, and a brief summary of the binary.
   - Tasks:
     - Explain what function does.
     - Identify vulnerability classes and suspicious or backdoor-like logic.
     - Assign structured risk data.
   - Output (strict JSON):
     - `function_summary`
     - `risk_score`
     - `issue_list`: each with `category`, `cwe`, `description`, `evidence`, `exploitability`, `confidence`.

3. **Cross-function reasoning (contextual pass)**
   - For example:
     - Handler function that dispatches commands.
     - Auth function that consults multiple backends.
   - Input:
     - Condensed summaries from multiple related functions.
   - Tasks:
     - Detect hidden command paths (magic strings, debug commands).
     - Identify secret-dependent flows across function boundaries (e.g., unnecessary data exfil).
   - Output:
     - Higher-level findings tying multiple functions together.

4. **Binary-level summary**
   - Input:
     - Aggregated high-risk findings.
   - Output:
     - “Executive” summary of binary behavior and risks.
     - High-severity flags specifically for backdoors/Trojans vs accidental bugs.

#### Stage 4 – Validation, guardrails, and human-in-the-loop

- Sanity checks:
  - Ensure findings reference real addresses or functions.
  - Cross-check with existing static analysis where possible (e.g., confirm a mentioned string address actually exists).
- Guardrails:
  - Reject or flag outputs that:
    - Don’t match the schema.
    - Repeat known hallucination patterns (e.g., referencing missing functions).
- Human review:
  - Analysts review top N highest-risk findings per binary.
  - Analysts can “confirm”, “reject”, or “needs more investigation”.
  - Feed this back as labeled data to tune prompts or fine-tune smaller internal models.

#### Stage 5 – Storage, indexing, and APIs

- Database schema:
  - `binaries`: metadata, hashes, build lineage.
  - `functions`: symbol info, addresses, static features.
  - `llm_findings`: each with risk scores, issue type, evidence, timestamp, model version.
- Embeddings:
  - Embed representative code snippets / decompiled functions.
  - Use vector search for:
    - “Find functions similar to this known backdoor.”
    - “Show me all functions that look like command handlers with hidden options.”
- APIs:
  - REST/GraphQL endpoints to:
    - Query findings for a specific build.
    - Trigger re-analysis on new model versions.
  - CI integration:
    - Fail build if new high-severity backdoor-like findings appear vs previous baseline.

### 2.4. Scaling and quality considerations

- **Cost control**
  - Heavy gating: only send top X% suspicious functions to the largest model.
  - Cache analyses by content hash (don’t reanalyze identical functions).
  - Use smaller, cheaper models for early tiers (triage, classification).

- **Latency**
  - Work queue + workers architecture.
  - Batch LLM calls where API supports it (process multiple chunks in one call).

- **Quality & evolution**
  - Maintain a test corpus:
    - Known Trojans and backdoors.
    - Clean binaries.
  - Regression tests:
    - New prompt/model versions must not perform worse on this corpus.

- **Security / data handling**
  - Decide data residency (on-prem vs cloud).
  - Redact user-specific data / secrets before sending to LLM if needed.

### 2.5. Distilled but powerful PoC version

From the “top tier” design, here is a realistic distilled subset you can build now (and later grow):

**Keep (for PoC):**

- Single-binary focus (no full firmware unpacking).
- Static tools:
  - `objdump`, `strings`, `readelf`; optional basic Ghidra/Rizin export.
- Single-stage LLM analysis (function-level) + one aggregation stage:
  - Use gemini-3-pro-preview only.
- Simple heuristics:
  - Score functions by presence of interesting strings and size.
- Storage:
  - JSON files + maybe a SQLite DB.

**Defer (for later “industry” build):**

- Full firmware unpacking and multi-binary correlation.
- Sophisticated CFG-based path reasoning and symbolic execution.
- Multi-model gating with cheaper vs expensive models.
- Vector databases and similarity search.
- CI/CD and enterprise dashboards.

**Concrete distilled architecture:**

1. `extract_binary_info.py`
   - Run tools, parse functions and strings.

2. `heuristics.py`
   - Compute per-function static interest scores (size + strings + call count).

3. `llm_client.py`
   - Single function: `analyze_function(chunk) -> FunctionReport`.

4. `analyzer.py`
   - Orchestrate:
     - Select top N functions by interest score.
     - Analyze each with Gemini.
     - Save `function_reports.json`.

5. `aggregate.py`
   - Feed top suspicious functions’ summaries into Gemini.
   - Produce `binary_report.md` with:
     - Overview.
     - Top suspicious functions.
     - Backdoor/Trojan risk assessment.

Once the Jetson arrives, we can:

- Reuse this exact pipeline on Jetson-specific binaries.
- Gradually enrich extraction (e.g., Ghidra IR, firmware images).
- Couple the static analysis with your dynamic telemetry findings to produce a combined “supply-chain anomaly” story.

