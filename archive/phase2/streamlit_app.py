"""Streamlit UI for Phase 2 static LLM binary analysis.

This app wraps run_pipeline() from analyze_binary.py to provide:
- File upload and sample binaries (clean/trojan) selection.
- Parameter controls (model, max functions, chunk size, risk threshold).
- Live progress updates while extraction/analysis runs.
- Rich visualizations of risk distribution, top functions, categories, and IoCs.
- Rendering of the generated markdown executive report and per-function findings.
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from analyze_binary import FunctionReport, run_pipeline

REPO_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = REPO_ROOT / "reports"
SAMPLE_BINARIES = [
    ("Trojan sample", REPO_ROOT / "firmware_trojan"),
    ("Clean sample", REPO_ROOT / "firmware_clean"),
]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_reports(path: Path) -> tuple[dict[str, Any], list[FunctionReport]]:
    raw = _load_json(path)
    # Pydantic will coerce missing optional fields gracefully
    model = [FunctionReport.model_validate(obj) for obj in raw.get("function_reports", [])]
    return raw, model


def _list_existing_reports() -> list[Path]:
    if not REPORTS_DIR.exists():
        return []
    return sorted([p for p in REPORTS_DIR.iterdir() if p.is_dir()])


def _function_df(reports: list[FunctionReport]) -> pd.DataFrame:
    rows = []
    for r in reports:
        rows.append(
            {
                "function": r.function_name,
                "risk": r.risk_score,
                "suspicious": r.suspicious,
                "security_relevant": r.security_relevant,
                "confidence": r.confidence,
                "categories": ", ".join(r.categories),
                "indicators": ", ".join(r.indicators),
                "c2": ", ".join(r.c2_indicators),
                "stealth": ", ".join(r.stealth_indicators),
                "network_actions": ", ".join(r.network_actions),
                "data_types": ", ".join(r.data_types_accessed),
                "recommended": ", ".join(r.recommended_actions),
                "summary": r.summary,
            }
        )
    return pd.DataFrame(rows)


def _category_counts(reports: list[FunctionReport]) -> pd.DataFrame:
    counter: dict[str, int] = {}
    for r in reports:
        for c in r.categories:
            counter[c] = counter.get(c, 0) + 1
    return pd.DataFrame({"category": list(counter.keys()), "count": list(counter.values())})


def _render_markdown(md_path: Path) -> None:
    st.markdown(md_path.read_text(encoding="utf-8"))


def _progress_hook():
    status = st.empty()

    def hook(phase: str, detail: str) -> None:
        status.info(f"{phase}: {detail}")

    return hook


def _run_analysis(
    *,
    binary_path: Path,
    model: str,
    max_functions: int,
    batch_size: int,
    max_lines_per_chunk: int,
    risk_threshold: float,
    skip_llm: bool,
    force: bool,
) -> dict[str, Any]:
    return run_pipeline(
        binary=binary_path,
        model=model,
        outdir=REPORTS_DIR,
        tmpdir=REPO_ROOT / "tmp",
        max_functions=max_functions,
        batch_size=batch_size,
        max_lines_per_chunk=max_lines_per_chunk,
        risk_threshold=risk_threshold,
        skip_llm=skip_llm,
        force=force,
        progress_callback=_progress_hook(),
    )


# --- UI layout ---
st.set_page_config(
    page_title="Phase 2 Static Binary Analyzer",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ Phase 2 Static LLM Binary Analysis UI")
st.caption(
    "Hybrid static + LLM-assisted firmware triage. Upload an ELF, run Cerebras zai-glm-4.6, and inspect risk triage with visuals."
)

with st.expander("About the pipeline", expanded=False):
    st.write(
        """
        - Extraction: objdump/readelf/nm/strings + file metadata
        - Chunking: function-first, section fallback for stripped binaries
        - LLM: Cerebras `zai-glm-4.6` with structured JSON schema (risk, categories, IoCs, recommendations)
        - Outputs: JSON per-function, markdown executive summary, cached under `phase2/reports/`
        """
    )

# Sidebar controls
st.sidebar.header("Run settings")
uploaded = st.sidebar.file_uploader("Upload ELF binary", type=None)
use_sample = st.sidebar.selectbox(
    "Or pick a sample", ["None"] + [label for label, _ in SAMPLE_BINARIES]
)
model = st.sidebar.text_input("Model", value="zai-glm-4.6")
max_functions = st.sidebar.slider("Max functions", 5, 80, 30)
batch_size = st.sidebar.slider("Batch size", 1, 16, 8)
max_lines = st.sidebar.slider("Max lines per chunk", 100, 600, 350, step=25)
risk_threshold = st.sidebar.slider("Risk threshold", 0.0, 1.0, 0.4, 0.05)
skip_llm = st.sidebar.checkbox("Skip LLM (extraction only)", value=False)
force = st.sidebar.checkbox("Force re-extraction", value=False)

st.sidebar.markdown("---")
st.sidebar.write("Existing report folders:")
for rep in _list_existing_reports():
    st.sidebar.write(f"- {rep.name}")

run_btn = st.sidebar.button("Run analysis", type="primary")

# Main content columns
col_left, col_right = st.columns([2, 1])

analysis_result: Optional[dict[str, Any]] = None
function_reports: list[FunctionReport] = []
binary_md_path: Optional[Path] = None
function_json_path: Optional[Path] = None

if run_btn:
    binary_path: Optional[Path] = None
    if uploaded is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(uploaded.getbuffer())
        tmp.flush()
        binary_path = Path(tmp.name)
    elif use_sample != "None":
        for label, path in SAMPLE_BINARIES:
            if label == use_sample:
                binary_path = path
                break
    else:
        st.error("Provide an uploaded ELF or choose a sample.")

    if binary_path:
        with st.spinner("Running analysis..."):
            try:
                analysis_result = _run_analysis(
                    binary_path=binary_path,
                    model=model,
                    max_functions=max_functions,
                    batch_size=batch_size,
                    max_lines_per_chunk=max_lines,
                    risk_threshold=risk_threshold,
                    skip_llm=skip_llm,
                    force=force,
                )
                if not skip_llm:
                    _, function_reports = _load_reports(analysis_result["function_reports_path"])
                    binary_md_path = analysis_result["binary_report_path"]
                    function_json_path = analysis_result["function_reports_path"]
            except Exception as exc:
                st.error(f"Run failed: {exc}")

# When results are available, render them
if analysis_result and not skip_llm and function_reports:
    df = _function_df(function_reports)
    with col_left:
        st.subheader("Risk overview")
        high_risk = (df["risk"] >= risk_threshold).sum()
        st.metric("High-risk functions", high_risk)
        fig_hist = px.histogram(df, x="risk", nbins=20, title="Risk score distribution")
        st.plotly_chart(fig_hist, use_container_width=True)

        fig_top = px.bar(
            df.sort_values("risk", ascending=False).head(12),
            x="risk",
            y="function",
            orientation="h",
            title="Top suspicious functions",
        )
        st.plotly_chart(fig_top, use_container_width=True)

        cat_df = _category_counts(function_reports)
        if not cat_df.empty:
            fig_cat = px.bar(cat_df.sort_values("count", ascending=False), x="category", y="count", title="Behavior categories")
            st.plotly_chart(fig_cat, use_container_width=True)

    with col_right:
        st.subheader("Executive report")
        if binary_md_path:
            _render_markdown(binary_md_path)

    st.subheader("Per-function findings")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Detailed IoCs and recommendations
    with st.expander("Indicators and recommendations", expanded=False):
        iocs_rows = []
        for r in function_reports:
            iocs_rows.append(
                {
                    "function": r.function_name,
                    "indicators": ", ".join(r.indicators or []),
                    "c2": ", ".join(r.c2_indicators or []),
                    "stealth": ", ".join(r.stealth_indicators or []),
                    "network_actions": ", ".join(r.network_actions or []),
                    "recommended": ", ".join(r.recommended_actions or []),
                }
            )
        st.dataframe(pd.DataFrame(iocs_rows), use_container_width=True, hide_index=True)

    if function_json_path:
        st.download_button(
            label="Download function_reports.json",
            file_name=function_json_path.name,
            mime="application/json",
            data=function_json_path.read_bytes(),
        )

elif analysis_result and skip_llm:
    st.info("Extraction complete. LLM skipped; reports not generated.")

# Demo section: show hardest findings from existing reports if present
with st.expander("Demo: hardest findings from cached reports", expanded=False):
    existing = _list_existing_reports()
    if not existing:
        st.write("No cached reports yet. Run an analysis to populate this section.")
    else:
        for rep in existing:
            func_path = rep / "function_reports.json"
            md_path = rep / "binary_report.md"
            if func_path.exists():
                _, reports = _load_reports(func_path)
                df_demo = _function_df(reports)
                st.markdown(f"**{rep.name}**")
                st.dataframe(
                    df_demo.sort_values("risk", ascending=False).head(5),
                    use_container_width=True,
                    hide_index=True,
                )
            if md_path.exists():
                st.markdown("Executive summary:")
                _render_markdown(md_path)
                st.markdown("---")
