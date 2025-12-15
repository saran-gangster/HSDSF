#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load .env file from project root
load_dotenv()

console = Console()


INTERESTING_STRING_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"password",
        r"passwd",
        r"admin",
        r"root",
        r"debug",
        r"backdoor",
        r"secret",
        r"token",
        r"key",
        r"auth",
        r"login",
        r"shell",
        r"exec",
        r"system\(",
        r"getenv",
        r"socket",
        r"connect",
        r"bind",
        r"listen",
        r"curl",
        r"wget",
    ]
]

INTERESTING_CALLEE_SUBSTRINGS = [
    "getenv",
    "system",
    "popen",
    "execl",
    "execve",
    "fork",
    "vfork",
    "clone",
    "socket",
    "connect",
    "bind",
    "listen",
    "accept",
    "open",
    "fopen",
    "unlink",
]


class FunctionChunk(BaseModel):
    name: str
    start_addr: str
    disassembly: str
    strings_nearby: list[str] = Field(default_factory=list)


class Finding(BaseModel):
    type: str
    description: str
    evidence: str


class FunctionReport(BaseModel):
    function_name: str
    summary: str
    security_relevant: bool
    suspicious: bool
    risk_score: float
    findings: list[Finding] = Field(default_factory=list)
    notes: str = ""
    confidence: float = 0.5
    categories: list[str] = Field(default_factory=list)
    indicators: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    c2_indicators: list[str] = Field(default_factory=list)
    stealth_indicators: list[str] = Field(default_factory=list)
    data_types_accessed: list[str] = Field(default_factory=list)
    network_actions: list[str] = Field(default_factory=list)


class BinaryReport(BaseModel):
    binary_path: str
    binary_sha256: str
    metadata: dict[str, Any]
    notable_strings: list[str]
    analyzed_functions: int
    model: Optional[str] = None
    function_reports: list[FunctionReport]


@dataclass(frozen=True)
class ExtractionArtifacts:
    work_dir: Path
    dis_path: Path
    strings_path: Path
    meta_path: Path
    symbols_path: Path
    file_path: Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_tool_exists(tool: str) -> None:
    if shutil.which(tool) is None:
        raise RuntimeError(
            f"Missing required tool '{tool}'. Install it (e.g. via your distro package manager) and retry."
        )


def run_cmd(
    args: list[str],
    *,
    cwd: Optional[Path] = None,
    stdout_path: Optional[Path] = None,
) -> None:
    stdout_handle = None
    try:
        if stdout_path is not None:
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_handle = stdout_path.open("wb")
        subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            stdout=stdout_handle,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"Command failed: {' '.join(args)}\n{msg}") from e
    finally:
        if stdout_handle:
            stdout_handle.close()


def extract_artifacts(binary_path: Path, *, tmp_root: Path, force: bool) -> ExtractionArtifacts:
    ensure_tool_exists("objdump")
    ensure_tool_exists("readelf")
    ensure_tool_exists("nm")
    ensure_tool_exists("strings")
    ensure_tool_exists("file")

    binary_sha = sha256_file(binary_path)
    work_dir = tmp_root / f"{binary_path.name}_{binary_sha[:8]}"
    work_dir.mkdir(parents=True, exist_ok=True)

    dis_path = work_dir / f"{binary_path.name}.dis"
    strings_path = work_dir / f"{binary_path.name}.strings"
    meta_path = work_dir / f"{binary_path.name}.meta"
    symbols_path = work_dir / f"{binary_path.name}.symbols"
    file_path = work_dir / f"{binary_path.name}.file"

    def missing(p: Path) -> bool:
        return force or (not p.exists()) or p.stat().st_size == 0

    if missing(dis_path):
        run_cmd(["objdump", "-d", "-Mintel", "-w", str(binary_path)], stdout_path=dis_path)
    if missing(strings_path):
        run_cmd(["strings", "-t", "x", str(binary_path)], stdout_path=strings_path)
    if missing(meta_path):
        run_cmd(["readelf", "-h", "-S", "-s", str(binary_path)], stdout_path=meta_path)
    if missing(symbols_path):
        run_cmd(["nm", "-C", str(binary_path)], stdout_path=symbols_path)
    if missing(file_path):
        run_cmd(["file", str(binary_path)], stdout_path=file_path)

    return ExtractionArtifacts(
        work_dir=work_dir,
        dis_path=dis_path,
        strings_path=strings_path,
        meta_path=meta_path,
        symbols_path=symbols_path,
        file_path=file_path,
    )


def parse_file_output(text: str) -> str:
    return text.strip()


def parse_arch_from_readelf(meta_text: str) -> Optional[str]:
    for line in meta_text.splitlines():
        if "Machine:" in line:
            return line.split("Machine:", 1)[1].strip()
    return None


_STRING_LINE_RE = re.compile(r"^\s*([0-9a-fA-F]+)\s+(.*)$")


def parse_strings(strings_text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for line in strings_text.splitlines():
        m = _STRING_LINE_RE.match(line)
        if not m:
            continue
        addr_hex, s = m.group(1), m.group(2)
        try:
            addr = int(addr_hex, 16)
        except ValueError:
            continue
        s = s.strip()
        if not s:
            continue
        out.append((addr, s))
    return out


def select_notable_strings(strings: list[tuple[int, str]], *, limit: int = 80) -> list[str]:
    scored: list[tuple[int, str]] = []
    for _, s in strings:
        score = 0
        for pat in INTERESTING_STRING_PATTERNS:
            if pat.search(s):
                score += 3
        if len(s) >= 8:
            score += 1
        if score > 0:
            scored.append((score, s))

    scored.sort(key=lambda t: (t[0], len(t[1])), reverse=True)

    seen: set[str] = set()
    result: list[str] = []
    for _, s in scored:
        if s in seen:
            continue
        seen.add(s)
        result.append(s)
        if len(result) >= limit:
            break
    return result


_FUNC_HEADER_RE = re.compile(r"^\s*([0-9a-fA-F]+)\s+<([^>]+)>:\s*$")


def parse_objdump_functions(dis_text: str) -> list[tuple[str, str, list[str]]]:
    """Returns list of (func_name, start_addr_hex, lines).

    Note: stripped binaries may have very few symbol headers, so callers should
    consider a fallback chunking strategy.
    """
    functions: list[tuple[str, str, list[str]]] = []
    cur_name: Optional[str] = None
    cur_addr: Optional[str] = None
    cur_lines: list[str] = []

    def flush() -> None:
        nonlocal cur_name, cur_addr, cur_lines
        if cur_name is not None and cur_addr is not None and cur_lines:
            functions.append((cur_name, cur_addr, cur_lines))
        cur_name, cur_addr, cur_lines = None, None, []

    for line in dis_text.splitlines():
        m = _FUNC_HEADER_RE.match(line)
        if m:
            flush()
            cur_addr = m.group(1)
            cur_name = m.group(2)
            cur_lines = [line]
            continue
        if cur_name is not None:
            cur_lines.append(line)

    flush()
    return functions


_SECTION_HEADER_RE = re.compile(r"^Disassembly of section (\S+):\s*$")


def parse_objdump_sections(dis_text: str) -> list[tuple[str, list[str]]]:
    """Returns list of (section_name, lines) for each disassembled section."""
    sections: list[tuple[str, list[str]]] = []
    cur_name: Optional[str] = None
    cur_lines: list[str] = []

    def flush() -> None:
        nonlocal cur_name, cur_lines
        if cur_name is not None and cur_lines:
            sections.append((cur_name, cur_lines))
        cur_name, cur_lines = None, []

    for line in dis_text.splitlines():
        m = _SECTION_HEADER_RE.match(line)
        if m:
            flush()
            cur_name = m.group(1)
            cur_lines = [line]
            continue
        if cur_name is not None:
            cur_lines.append(line)

    flush()
    return sections


_INSN_ADDR_RE = re.compile(r"^\s*([0-9a-fA-F]+):\s")


def first_instruction_addr_hex(lines: list[str], default: str) -> str:
    for line in lines:
        m = _INSN_ADDR_RE.match(line)
        if m:
            return m.group(1)
    return default


def chunk_functions(
    functions: list[tuple[str, str, list[str]]],
    *,
    max_lines_per_chunk: int,
    strings_nearby: list[str],
) -> list[FunctionChunk]:
    chunks: list[FunctionChunk] = []
    for name, start_addr, lines in functions:
        if len(lines) < 6:
            continue
        if name.endswith("@plt") or name == ".plt":
            continue

        if len(lines) <= max_lines_per_chunk:
            chunks.append(
                FunctionChunk(
                    name=name,
                    start_addr=start_addr,
                    disassembly="\n".join(lines),
                    strings_nearby=strings_nearby,
                )
            )
            continue

        seg_idx = 1
        for i in range(0, len(lines), max_lines_per_chunk):
            seg_lines = lines[i : i + max_lines_per_chunk]
            seg_addr = first_instruction_addr_hex(seg_lines, start_addr)
            chunks.append(
                FunctionChunk(
                    name=f"{name}#{seg_idx}",
                    start_addr=seg_addr,
                    disassembly="\n".join(seg_lines),
                    strings_nearby=strings_nearby,
                )
            )
            seg_idx += 1

    return chunks


def chunk_regions(
    regions: list[tuple[str, list[str]]],
    *,
    max_lines_per_chunk: int,
    strings_nearby: list[str],
) -> list[FunctionChunk]:
    chunks: list[FunctionChunk] = []
    for region_name, lines in regions:
        # Drop tiny regions; they tend to be headings/blank areas.
        if len(lines) < 20:
            continue

        seg_idx = 1
        for i in range(0, len(lines), max_lines_per_chunk):
            seg_lines = lines[i : i + max_lines_per_chunk]
            if len(seg_lines) < 20:
                continue
            seg_addr = first_instruction_addr_hex(seg_lines, default="0")
            chunks.append(
                FunctionChunk(
                    name=f"{region_name}#{seg_idx}",
                    start_addr=seg_addr,
                    disassembly="\n".join(seg_lines),
                    strings_nearby=strings_nearby,
                )
            )
            seg_idx += 1
    return chunks


def interest_score(chunk: FunctionChunk, notable_strings: list[str]) -> float:
    score = 0.0
    lines = chunk.disassembly.splitlines()
    score += min(len(lines) / 200.0, 3.0)

    hay = chunk.disassembly.lower()
    for sub in INTERESTING_CALLEE_SUBSTRINGS:
        if sub in hay:
            score += 1.5

    for s in notable_strings[:40]:
        if s and s.lower() in hay:
            score += 2.0

    name = chunk.name.lower()
    for kw in ["auth", "login", "passwd", "password", "dispatch", "handle", "cmd"]:
        if kw in name:
            score += 1.0

    return score


def extract_json_object(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: grab first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        return json.loads(candidate)

    raise ValueError("No JSON object found in response")


class CerebrasClient:
    def __init__(self, *, api_key: str, model: str = "zai-glm-4.6", batch_size: int = 8):
        from cerebras.cloud.sdk import Cerebras

        self._client = Cerebras(api_key=api_key)
        self._model = model
        self._batch_size = batch_size
        self._call_count = 0
        self._last_call_time = 0.0

    def analyze_function(
        self,
        *,
        chunk: FunctionChunk,
        binary_name: str,
        arch: str,
        file_summary: str,
        notable_strings: list[str],
    ) -> FunctionReport:
        types = self._types

        prompt = "\n".join(
            [
                "[BINARY CONTEXT]",
                f"Binary name: {binary_name}",
                f"Architecture: {arch}",
                f"File: {file_summary}",
                f"Notable global strings: {notable_strings[:40]}",
                "",
                "[FUNCTION DISASSEMBLY]",
                f"Function: {chunk.name}",
                f"Start address: {chunk.start_addr}",
                "",
                chunk.disassembly,
                "",
                "[OPTIONAL STRINGS NEAR THIS FUNCTION]",
                "\n".join(chunk.strings_nearby[:40]) if chunk.strings_nearby else "(none)",
                "",
                "[INSTRUCTIONS]",
                "1. Summarize what this function likely does.",
                "2. Identify any clearly security-relevant operations.",
                "3. Flag any behavior that could be a backdoor or Trojan trigger.",
                "4. Highlight any magic constants, suspicious strings, or hidden branches.",
                "5. Provide a numeric risk_score between 0 and 1 (0 = benign, 1 = clearly malicious).",
                "6. Provide specific evidence: line snippets or addresses.",
                "7. Include categories (e.g., backdoor, C2, exfiltration, persistence).",
                "8. Include indicators (IoCs), network_actions, data_types_accessed, stealth_indicators, c2_indicators.",
                "9. Add recommended_actions for an analyst.",
                "",
                "Return JSON with keys:",
                '- "function_name": str',
                '- "summary": str',
                '- "security_relevant": bool',
                '- "suspicious": bool',
                '- "risk_score": float',
                '- "findings": [ { "type": str, "description": str, "evidence": str } ]',
                '- "notes": str',
                '- "confidence": float',
                '- "categories": [str]',
                '- "indicators": [str]',
                '- "recommended_actions": [str]',
                '- "c2_indicators": [str]',
                '- "stealth_indicators": [str]',
                '- "data_types_accessed": [str]',
                '- "network_actions": [str]',
            ]
        )

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]

        config = types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.95,
            max_output_tokens=2048,
            response_mime_type="application/json",
        )

        resp = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        obj = extract_json_object(resp.text or "")
        return FunctionReport.model_validate(obj)

    def analyze_batch(
        self,
        *,
        chunks: list[FunctionChunk],
        binary_name: str,
        arch: str,
        file_summary: str,
        notable_strings: list[str],
    ) -> list[FunctionReport]:
        """Analyze multiple functions in one API call with structured JSON output."""

        # Build a consolidated prompt with all functions
        function_blocks = []
        for idx, chunk in enumerate(chunks, start=1):
            block = "\n".join(
                [
                    f"--- FUNCTION {idx} ---",
                    f"Name: {chunk.name}",
                    f"Start address: {chunk.start_addr}",
                    "",
                    chunk.disassembly[:3000],  # Cerebras has 40k token limit
                    "",
                ]
            )
            function_blocks.append(block)

        system_prompt = "You are a security analyst specialized in reverse engineering firmware binaries. Analyze disassembled functions and identify suspicious patterns, backdoors, and security-relevant behavior."

        user_prompt = "\n".join(
            [
                "[BINARY CONTEXT]",
                f"Binary name: {binary_name}",
                f"Architecture: {arch}",
                f"File: {file_summary}",
                f"Notable global strings: {notable_strings[:40]}",
                "",
                "[FUNCTIONS TO ANALYZE]",
                "\n".join(function_blocks),
                "",
                "[INSTRUCTIONS]",
                "For each function:",
                "1. Summarize what it likely does.",
                "2. Identify security-relevant operations (auth, network, crypto, file I/O, process control).",
                "3. Flag backdoor/Trojan-like behavior (magic values, hidden branches, suspicious env checks).",
                "4. Highlight magic constants, suspicious strings, or conditional bypasses.",
                "5. Assign risk_score: 0.0 (benign) to 1.0 (clearly malicious) and confidence 0-1.",
                "6. Provide specific evidence (addresses, instructions, strings).",
                "7. Add categories (backdoor, persistence, credential access, network/C2, exfiltration, anti-analysis).",
                "8. Include indicators/IoCs, c2_indicators, stealth_indicators, data_types_accessed, network_actions.",
                "9. Recommend analyst actions (triage, instrumentation, patching).",
            ]
        )

        # Define structured output schema
        class FunctionReportList(BaseModel):
            reports: list[FunctionReport]

        # Rate limit handling with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Basic rate limiting: wait 1s between calls
                elapsed = time.time() - self._last_call_time
                if elapsed < 1.0:
                    time.sleep(1.0 - elapsed)

                resp = self._client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model=self._model,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "function_analysis",
                            "schema": FunctionReportList.model_json_schema(),
                        },
                    },
                    max_completion_tokens=16000,
                    temperature=0.3,
                    top_p=0.95,
                )
                self._last_call_time = time.time()
                self._call_count += 1

                content = resp.choices[0].message.content
                if not content:
                    raise ValueError("Empty response from API")

                result = FunctionReportList.model_validate_json(content)
                return result.reports

            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "quota" in err_str:
                    wait = min(2 ** attempt * 2, 30)
                    console.print(f"[yellow]Rate limit hit, waiting {wait}s...[/yellow]")
                    time.sleep(wait)
                    continue
                else:
                    console.print(f"[red]Batch analysis error: {e}[/red]")
                    return [
                        FunctionReport(
                            function_name=chunk.name,
                            summary="Batch LLM call failed",
                            security_relevant=False,
                            suspicious=False,
                            risk_score=0.0,
                            findings=[Finding(type="error", description=str(e), evidence=chunk.start_addr)],
                            notes="",
                        )
                        for chunk in chunks
                    ]

        # Exhausted retries
        return [
            FunctionReport(
                function_name=chunk.name,
                summary="Rate limit exceeded",
                security_relevant=False,
                suspicious=False,
                risk_score=0.0,
                findings=[Finding(type="error", description="Max retries", evidence=chunk.start_addr)],
                notes="",
            )
            for chunk in chunks
        ]

    def aggregate(
        self,
        *,
        binary_name: str,
        arch: str,
        file_summary: str,
        notable_strings: list[str],
        top_reports: list[FunctionReport],
    ) -> str:
        compact = [
            {
                "function_name": r.function_name,
                "risk_score": r.risk_score,
                "summary": r.summary,
                "findings": [f.model_dump() for f in r.findings[:5]],
            }
            for r in top_reports[:12]
        ]

        system_prompt = "You are a security analyst specialized in reverse engineering firmware binaries. Produce concise, actionable markdown reports."

        user_prompt = "\n".join(
            [
                "Given LLM function-level findings, produce a concise markdown report.",
                "",
                "[BINARY CONTEXT]",
                f"Binary name: {binary_name}",
                f"Architecture: {arch}",
                f"File: {file_summary}",
                f"Notable global strings: {notable_strings[:50]}",
                "",
                "[TOP SUSPICIOUS FUNCTIONS]",
                json.dumps(compact, indent=2),
                "",
                "[OUTPUT FORMAT]",
                "Return markdown with:",
                "- Overall verdict (Backdoor/Trojan likelihood: Low/Medium/High)",
                "- Top suspicious functions (bullets with function name, risk score, and evidence)",
                "- Suggested follow-up manual RE steps",
            ]
        )

        resp = self._client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self._model,
            max_completion_tokens=4096,
            temperature=0.3,
            top_p=0.95,
        )

        return (resp.choices[0].message.content or "").strip()


def run_pipeline(
    *,
    binary: Path,
    model: str = "zai-glm-4.6",
    outdir: Path | str = Path("phase2") / "reports",
    tmpdir: Path | str = Path("phase2") / "tmp",
    max_functions: int = 40,
    batch_size: int = 8,
    max_lines_per_chunk: int = 350,
    risk_threshold: float = 0.4,
    skip_llm: bool = False,
    force: bool = False,
    progress_callback: Optional[Callable[[str, str], None]] = None,
) -> dict[str, Any]:
    """Run the Phase 2 pipeline programmatically (CLI-compatible).

    Returns a dict with paths and parsed data suitable for the Streamlit UI.
    """

    def notify(phase: str, detail: str) -> None:
        if progress_callback:
            progress_callback(phase, detail)

    binary_path = Path(binary).expanduser().resolve()
    if not binary_path.exists():
        raise FileNotFoundError(f"Binary not found: {binary_path}")

    tmp_root = Path(tmpdir).expanduser().resolve()
    out_root = Path(outdir).expanduser().resolve()

    binary_sha = sha256_file(binary_path)
    report_dir = out_root / f"{binary_path.name}_{binary_sha[:8]}"
    report_dir.mkdir(parents=True, exist_ok=True)

    notify("extract", "Extracting artifacts")
    if progress_callback:
        artifacts = extract_artifacts(binary_path, tmp_root=tmp_root, force=force)
    else:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
            t = progress.add_task("Extracting artifacts", total=None)
            artifacts = extract_artifacts(binary_path, tmp_root=tmp_root, force=force)
            progress.update(t, completed=1)

    dis_text = artifacts.dis_path.read_text("utf-8", errors="replace")
    strings_text = artifacts.strings_path.read_text("utf-8", errors="replace")
    meta_text = artifacts.meta_path.read_text("utf-8", errors="replace")
    file_text = artifacts.file_path.read_text("utf-8", errors="replace")

    arch = parse_arch_from_readelf(meta_text) or "unknown"
    file_summary = parse_file_output(file_text)

    all_strings = parse_strings(strings_text)
    notable_strings = select_notable_strings(all_strings)

    functions = parse_objdump_functions(dis_text)
    if len(functions) >= 10:
        chunks = chunk_functions(
            functions,
            max_lines_per_chunk=max_lines_per_chunk,
            strings_nearby=notable_strings,
        )
    else:
        sections = parse_objdump_sections(dis_text)
        if sections:
            chunks = chunk_regions(
                sections,
                max_lines_per_chunk=max_lines_per_chunk,
                strings_nearby=notable_strings,
            )
        else:
            chunks = chunk_regions(
                [("disassembly", dis_text.splitlines())],
                max_lines_per_chunk=max_lines_per_chunk,
                strings_nearby=notable_strings,
            )

    scored = [(interest_score(c, notable_strings), c) for c in chunks]
    scored.sort(key=lambda t: t[0], reverse=True)
    selected = [c for _, c in scored[: max(max_functions, 0)]]

    extraction_summary = {
        "binary": str(binary_path),
        "sha256": binary_sha,
        "arch": arch,
        "file": file_summary,
        "artifacts": {
            "work_dir": str(artifacts.work_dir),
            "dis": str(artifacts.dis_path),
            "strings": str(artifacts.strings_path),
            "meta": str(artifacts.meta_path),
            "symbols": str(artifacts.symbols_path),
            "file": str(artifacts.file_path),
        },
        "function_count": len(functions),
        "chunk_count": len(chunks),
        "selected_chunk_count": len(selected),
        "notable_string_count": len(notable_strings),
    }

    extraction_path = report_dir / "extraction_summary.json"
    extraction_path.write_text(json.dumps(extraction_summary, indent=2), encoding="utf-8")

    if skip_llm:
        return {
            "report_dir": report_dir,
            "extraction_summary_path": extraction_path,
            "function_reports_path": None,
            "binary_report_path": None,
            "binary_report": None,
            "suspicious": [],
        }

    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY is not set. Run with --skip-llm or set it in .env")

    client = CerebrasClient(api_key=api_key, model=model, batch_size=batch_size)

    reports: list[FunctionReport] = []

    if progress_callback:
        notify("analyze", f"Analyzing {len(selected)} functions")
        for i in range(0, len(selected), batch_size):
            batch = selected[i : i + batch_size]
            batch_reports = client.analyze_batch(
                chunks=batch,
                binary_name=binary_path.name,
                arch=arch,
                file_summary=file_summary,
                notable_strings=notable_strings,
            )
            reports.extend(batch_reports)
            notify("analyze", f"Completed batch {(i // batch_size) + 1}")
    else:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
            num_batches = (len(selected) + batch_size - 1) // batch_size
            task = progress.add_task(
                f"Analyzing {len(selected)} functions in {num_batches} batches", total=num_batches
            )
            for i in range(0, len(selected), batch_size):
                batch = selected[i : i + batch_size]
                batch_reports = client.analyze_batch(
                    chunks=batch,
                    binary_name=binary_path.name,
                    arch=arch,
                    file_summary=file_summary,
                    notable_strings=notable_strings,
                )
                reports.extend(batch_reports)
                progress.advance(task)

    binary_report_obj = BinaryReport(
        binary_path=str(binary_path),
        binary_sha256=binary_sha,
        metadata={"arch": arch, "file": file_summary},
        notable_strings=notable_strings,
        analyzed_functions=len(selected),
        model=model,
        function_reports=reports,
    )

    function_reports_path = report_dir / "function_reports.json"
    function_reports_path.write_text(binary_report_obj.model_dump_json(indent=2), encoding="utf-8")

    suspicious = [r for r in reports if r.risk_score >= risk_threshold]
    suspicious.sort(key=lambda r: r.risk_score, reverse=True)

    md = client.aggregate(
        binary_name=binary_path.name,
        arch=arch,
        file_summary=file_summary,
        notable_strings=notable_strings,
        top_reports=suspicious,
    )

    binary_report_path = report_dir / "binary_report.md"
    binary_report_path.write_text(md + "\n", encoding="utf-8")

    return {
        "report_dir": report_dir,
        "extraction_summary_path": extraction_path,
        "function_reports_path": function_reports_path,
        "binary_report_path": binary_report_path,
        "binary_report": binary_report_obj,
        "suspicious": suspicious,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Phase 2: static LLM-assisted binary analysis")
    parser.add_argument("binary", type=str, help="Path to binary (ELF preferred)")
    parser.add_argument("--model", type=str, default="zai-glm-4.6", help="Cerebras model to use")
    parser.add_argument("--outdir", type=str, default=str(Path("phase2") / "reports"))
    parser.add_argument("--tmpdir", type=str, default=str(Path("phase2") / "tmp"))
    parser.add_argument("--max-functions", type=int, default=40, help="Max functions to analyze")
    parser.add_argument("--batch-size", type=int, default=8, help="Functions per API call (leverages long context)")
    parser.add_argument("--max-lines-per-chunk", type=int, default=350)
    parser.add_argument("--risk-threshold", type=float, default=0.4)
    parser.add_argument("--skip-llm", action="store_true", help="Only run extraction + chunking")
    parser.add_argument("--force", action="store_true", help="Force re-run extraction tools")

    args = parser.parse_args(argv)

    try:
        result = run_pipeline(
            binary=Path(args.binary),
            model=args.model,
            outdir=Path(args.outdir),
            tmpdir=Path(args.tmpdir),
            max_functions=args.max_functions,
            batch_size=args.batch_size,
            max_lines_per_chunk=args.max_lines_per_chunk,
            risk_threshold=args.risk_threshold,
            skip_llm=args.skip_llm,
            force=args.force,
            progress_callback=None,
        )
    except Exception as exc:  # Keep CLI UX intact
        console.print(f"[red]{exc}[/red]")
        return 2

    if args.skip_llm:
        console.print(f"[green]Extraction complete.[/green] Wrote {result['extraction_summary_path']}")
    else:
        console.print(f"[green]Done.[/green] Report dir: {result['report_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
