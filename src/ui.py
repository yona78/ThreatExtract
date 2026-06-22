"""ThreatExtract — Streamlit presentation helpers (CSS, HTML, sidebar, engine loader)."""

from __future__ import annotations

import hashlib
import html
from pathlib import Path

import streamlit as st

from src.ner_engine import Entity, NerEngine

# assets/ lives at the project root (one level above this file's parent)
ASSETS = Path(__file__).resolve().parent.parent / "assets"


# --------------------------------------------------------------------------- #
# CSS injection
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    css = (ASSETS / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Color + label helpers
# --------------------------------------------------------------------------- #
def class_color(name: str) -> str:
    """Deterministic, distinct color per class name — works for any model."""
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    hue = int(digest[:6], 16) % 360
    return f"hsl({hue} 75% 62%)"


def section_label(text: str) -> None:
    st.markdown(f'<div class="te-label">{html.escape(text)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Header / chips / highlighted doc
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
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
