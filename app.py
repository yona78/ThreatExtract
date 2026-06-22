"""ThreatExtract — Streamlit UI for model-agnostic NER over threat-intel text."""

from __future__ import annotations

import hashlib
import html
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import load_config
from src.file_utils import ValidationError, read_upload, validate_text_file
from src.ner_engine import Entity, NerEngine

ASSETS = Path(__file__).parent / "assets"


# --------------------------------------------------------------------------- #
# Presentation helpers
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    css = (ASSETS / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def class_color(name: str) -> str:
    """Deterministic, distinct color per class name — works for any model."""
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    hue = int(digest[:6], 16) % 360
    return f"hsl({hue} 75% 62%)"


def section_label(text: str) -> None:
    st.markdown(f'<div class="te-label">{html.escape(text)}</div>', unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        '<div class="te-header">'
        '<div class="te-status"><span class="dot"></span>engine online</div>'
        '<div class="te-title"><span class="te-accent">Threat</span>Extract</div>'
        '<div class="te-tag">named-entity recognition for cyber-threat intelligence</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_class_chips(names_with_counts) -> None:
    chips = []
    for name, count in names_with_counts:
        color = class_color(name)
        suffix = f" <b>{count}</b>" if count is not None else ""
        chips.append(
            f'<span class="te-chip" style="--ent:{color}">{html.escape(name)}{suffix}</span>'
        )
    st.markdown('<div class="te-chips">' + "".join(chips) + "</div>", unsafe_allow_html=True)


def build_highlighted_html(text: str, entities: list[Entity]) -> str:
    parts: list[str] = ['<div class="te-doc">']
    cursor = 0
    for ent in sorted(entities, key=lambda e: (e.start, e.end)):
        if ent.start < cursor:  # skip a span already covered by a prior entity
            continue
        parts.append(html.escape(text[cursor : ent.start]))
        color = class_color(ent.class_name)
        tip = html.escape(f"{ent.class_name} · {ent.score:.0%}")
        parts.append(
            f'<span class="ent" style="--ent:{color}" title="{tip}">'
            f"{html.escape(text[ent.start : ent.end])}"
            f"<sup>{html.escape(ent.class_name)}</sup></span>"
        )
        cursor = ent.end
    parts.append(html.escape(text[cursor:]))
    parts.append("</div>")
    return "".join(parts)


def render_sidebar(engine: NerEngine, cfg) -> None:
    with st.sidebar:
        section_label("engine")
        rows = [
            ("model", Path(engine.model_path).name.replace("__", "/")),
            ("classes", str(len(engine.class_names))),
            ("max tokens", str(engine.max_length)),
            ("aggregation", engine.aggregation_strategy),
            ("max upload", f"{cfg.max_file_mb} MB"),
        ]
        meta = "".join(
            f'<div class="te-meta"><span class="k">{html.escape(k)}</span>'
            f'<span class="v">{html.escape(v)}</span></div>'
            for k, v in rows
        )
        st.markdown(meta, unsafe_allow_html=True)
        section_label("entity classes")
        render_class_chips([(c, None) for c in engine.class_names])


# --------------------------------------------------------------------------- #
# Engine loading (cached across reruns)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_engine(model_path: str, aggregation: str, overlap: int) -> NerEngine:
    return NerEngine(model_path, aggregation_strategy=aggregation, chunk_overlap=overlap)


def main() -> None:
    st.set_page_config(page_title="ThreatExtract", page_icon="🛡️", layout="wide")
    inject_css()
    render_header()

    cfg = load_config()
    try:
        with st.spinner("Loading NER model…"):
            engine = load_engine(cfg.model_path, cfg.aggregation_strategy, cfg.chunk_overlap)
    except Exception as exc:  # model missing / unreadable — guide the user
        st.error(
            f"Could not load a model from `{cfg.model_path}`.\n\n"
            "Run `python download_model.py` first, then point `MODEL_PATH` at the "
            f"printed directory.\n\nDetails: {exc}"
        )
        st.stop()

    render_sidebar(engine, cfg)

    section_label("upload a threat report (.txt)")
    upload = st.file_uploader(
        "Upload a .txt file",
        type=["txt"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if upload is None:
        st.markdown(
            '<div class="te-empty"><div class="big">Awaiting input</div>'
            "Drop a plain-text threat-intel report above to extract named entities."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # 1) Read fully into memory, then validate extension + real MIME + UTF-8.
    try:
        data = read_upload(upload, cfg.max_file_bytes)
        text = validate_text_file(upload.name, data)
    except ValidationError as exc:
        st.error(str(exc))
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("File", upload.name)
    col2.metric("Size", f"{len(data) / 1024:.1f} KB")
    col3.metric("Characters", f"{len(text):,}")

    # 2) Extract once per (file, model); reuse on later reruns so toggling the
    #    expander never re-runs inference. Progress bar shows on first compute.
    cache_key = (
        hashlib.md5(data).hexdigest(),
        cfg.model_path,
        cfg.aggregation_strategy,
        cfg.chunk_overlap,
    )
    if st.session_state.get("cache_key") != cache_key:
        st.toast("File validated", icon="✅")
        progress = st.progress(0.0, text="Analyzing…")

        def on_progress(done: int, total: int) -> None:
            progress.progress(done / total, text=f"Analyzing… chunk {done}/{total}")

        start = time.perf_counter()
        entities = engine.extract_entities(text, progress_cb=on_progress)
        elapsed = time.perf_counter() - start
        progress.empty()
        st.session_state.update(
            cache_key=cache_key, entities=entities, doc_text=text, elapsed=elapsed
        )
        st.toast(f"{len(entities)} entities extracted", icon="🛡️")

    entities: list[Entity] = st.session_state["entities"]
    text = st.session_state["doc_text"]
    elapsed = st.session_state["elapsed"]

    if not entities:
        st.info("No named entities were detected in this document.")
        return

    # 3) Summary.
    section_label("summary")
    m1, m2, m3 = st.columns([1, 1, 3])
    m1.metric("Entities", len(entities))
    m2.metric("Latency", f"{elapsed:.2f}s")
    with m3:
        counts = Counter(e.class_name for e in entities)
        render_class_chips(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    # 4) Required two-column results table + CSV download.
    section_label("identified entities")
    df = pd.DataFrame(
        [(e.class_name, e.text) for e in entities],
        columns=["Class Name", "Identified Entity"],
    )
    styled = df.style.map(
        lambda v: f"color: {class_color(v)}; font-weight: 600;",
        subset=["Class Name"],
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇  Download results (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"{Path(upload.name).stem}_entities.csv",
        mime="text/csv",
    )

    # 5) Highlighted document — the premium centerpiece.
    with st.expander("Highlighted document", expanded=True):
        st.markdown(build_highlighted_html(text, entities), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
