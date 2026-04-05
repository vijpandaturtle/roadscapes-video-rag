"""
app.py — Streamlit demo UI for the driving advice RAG system.

Run:
    streamlit run app.py
"""

import base64
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import anthropic
import cv2
import lancedb
import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from advise import (
    build_query_vector,
    generate_advice,
    get_device,
    load_clip as load_clip_model,
    retrieve_top_k_clips,
    sample_frames,
)


# ── Thumbnail helpers ─────────────────────────────────────────────────────────

def extract_thumbnail(video_path: str, frame_idx: int = 0) -> Image.Image | None:
    """Extract a single frame from a video file as a PIL image."""
    try:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx   = min(frame_idx, max(0, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception:
        return None


def frame_strip(frames: list[np.ndarray], max_w: int = 120) -> list[Image.Image]:
    """Resize a list of numpy frames to thumbnails."""
    thumbs = []
    for f in frames:
        img = Image.fromarray(f)
        ratio = max_w / img.width
        img   = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
        thumbs.append(img)
    return thumbs

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Roadscapes RAG Driving Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&family=Bebas+Neue&display=swap');

:root {
    --bg:      #0d0d0d;
    --surface: #141414;
    --border:  #222;
    --accent:  #c8ff00;
    --red:     #ff3c3c;
    --blue:    #4a9eff;
    --text:    #e8e8e8;
    --muted:   #555;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

[data-testid="stAppViewContainer"] {
    background-image:
        linear-gradient(rgba(200,255,0,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(200,255,0,0.03) 1px, transparent 1px);
    background-size: 32px 32px;
}

h1, h2, h3 { font-family: 'Bebas Neue', sans-serif !important; letter-spacing: 2px; }

[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--muted) !important;
    border-radius: 0 !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 18px !important;
    letter-spacing: 3px !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 12px 40px !important;
    width: 100% !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover    { opacity: 0.85 !important; }
.stButton > button:disabled { background: var(--muted) !important; }

.advice-card {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 24px 28px;
    margin-bottom: 12px;
}
.advice-card .tag  { font-size: 10px; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 10px; font-weight: 500; }
.advice-card .body { font-size: 15px; line-height: 1.7; color: var(--text); }

.card-action  { border-left: 3px solid var(--accent); }
.card-reason  { border-left: 3px solid var(--blue); }
.card-warning { border-left: 3px solid var(--red); }
.tag-action   { color: var(--accent); }
.tag-reason   { color: var(--blue); }
.tag-warning  { color: var(--red); }

.retrieval-item {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 14px 18px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.score-bar-wrap { flex: 1; height: 3px; background: var(--border); }
.score-bar-fill { height: 100%; background: var(--accent); }

.header-rule { border: none; border-top: 1px solid var(--border); margin: 4px 0 20px 0; }

[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 11px !important; letter-spacing: 2px !important; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Bebas Neue', sans-serif !important; font-size: 28px !important; }

footer, #MainMenu, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    device           = get_device()
    clip_model, proc = load_clip_model(device)
    client           = anthropic.Anthropic()
    return device, clip_model, proc, client


@st.cache_resource
def load_db():
    db_path = Path(__file__).parent / "index" / "roadscapes.lancedb"
    if not db_path.exists():
        st.error(f"LanceDB not found at {db_path}. Run ingest.py first.")
        st.stop()
    db    = lancedb.connect(str(db_path))
    table = db.open_table("frames_train")
    return table


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='font-size:52px;margin-bottom:0'>ROADSCAPES RAG DRIVING ASSISTANT</h1>", unsafe_allow_html=True)
st.markdown("<hr class='header-rule'>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#555;font-size:12px;letter-spacing:2px;margin-bottom:32px'>"
    "A Demo RAG Advisor built for the Roadscapes Video Dataset"
    "</p>",
    unsafe_allow_html=True,
)

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("#### INPUT")

    uploaded = st.file_uploader(
        "Drop a driving clip",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
    )

    top_k = st.slider("Retrieved clips (top-k)", min_value=1, max_value=5, value=3)

    run = st.button("ANALYSE CLIP", disabled=uploaded is None)

    if uploaded:
        st.video(uploaded)

with right:
    st.markdown("#### ADVICE")

    if not uploaded:
        st.markdown(
            "<div style='color:#333;font-size:13px;margin-top:40px'>"
            "Upload a clip on the left to generate driving advice."
            "</div>",
            unsafe_allow_html=True,
        )

# ── Inference ─────────────────────────────────────────────────────────────────
if run and uploaded:
    with right:
        device, clip_model, proc, client = load_models()
        table                            = load_db()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        with st.spinner("Embedding frames..."):
            query_vec = build_query_vector(
                clip_path=tmp_path,
                text=None,
                label=None,
                model=clip_model,
                processor=proc,
                device=device,
            )

        with st.spinner("Retrieving similar clips..."):
            retrieved = retrieve_top_k_clips(query_vec, table, top_k=top_k)

        with st.spinner("Generating advice..."):
            query_frames = sample_frames(tmp_path, n=5)
            advice = generate_advice(
                query_description=None,
                retrieved=retrieved,
                query_frames=query_frames,
                client=client,
            )

        os.unlink(tmp_path)

        # ── Advice cards ───────────────────────────────────────────────────
        st.markdown(f"""
        <div class="advice-card card-action">
            <div class="tag tag-action">▶ ACTION</div>
            <div class="body">{advice.get('action', '')}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="advice-card card-reason">
            <div class="tag tag-reason">◈ REASONING</div>
            <div class="body">{advice.get('reasoning', '')}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="advice-card card-warning">
            <div class="tag tag-warning">⚠ WARNING</div>
            <div class="body">{advice.get('warning', '')}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Query frame strip ──────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:10px;letter-spacing:3px;color:#555'>QUERY FRAMES</p>",
            unsafe_allow_html=True,
        )
        if query_frames:
            thumbs = frame_strip(query_frames, max_w=130)
            st.image(thumbs, width=130)

        # ── Metrics ────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("CLIPS RETRIEVED", top_k)
        m2.metric("TOP SIMILARITY", f"{retrieved[0]['score']:.2f}" if retrieved else "—")
        m3.metric("MODEL", "claude-opus-4-5")

        # ── Retrieved clips with thumbnails ───────────────────────────────
        st.markdown(
            "<br><p style='font-size:10px;letter-spacing:3px;color:#555'>RETRIEVED CONTEXT</p>",
            unsafe_allow_html=True,
        )

        for i, r in enumerate(retrieved, 1):
            score      = r["score"]
            name       = r["meta"]["clip_name"]
            clip_path  = r["meta"].get("clip_path", "")
            frame_idx  = r["meta"].get("frame_idx", 0)
            label_text = r["meta"].get("label", {}).get(
                "What is the action being performed by the ego vehicle? Answer in a single sentence.", ""
            )
            justif = r["meta"].get("label", {}).get(
                "What is the justification for the current action being performed by the ego vehicle? Answer in a single sentence.", ""
            )
            surroundings = r["meta"].get("label", {}).get(
                "Tell me about the surroundings such as the weather type, road type, time of day and scenario in two sentences.", ""
            )
            pct = int(score * 100)

            # try to load thumbnail from clip_path
            thumb = None
            if clip_path and Path(clip_path).exists():
                thumb = extract_thumbnail(clip_path, frame_idx=frame_idx)

            with st.expander(f"#{i}  {name}  —  {score:.3f}", expanded=(i == 1)):
                if thumb:
                    tcol, mcol = st.columns([1, 2], gap="small")
                    with tcol:
                        st.image(thumb, use_container_width=True)
                    with mcol:
                        st.markdown(
                            f"<p style='font-size:10px;letter-spacing:2px;color:#555;margin-bottom:4px'>ACTION</p>"
                            f"<p style='font-size:12px;color:#e8e8e8;line-height:1.6'>{label_text}</p>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<p style='font-size:10px;letter-spacing:2px;color:#555;margin-bottom:4px;margin-top:12px'>JUSTIFICATION</p>"
                            f"<p style='font-size:12px;color:#888;line-height:1.6'>{justif}</p>",
                            unsafe_allow_html=True,
                        )
                else:
                    # no thumbnail available — text only
                    st.markdown(
                        f"<p style='font-size:10px;letter-spacing:2px;color:#555;margin-bottom:4px'>ACTION</p>"
                        f"<p style='font-size:12px;color:#e8e8e8;line-height:1.6'>{label_text}</p>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<p style='font-size:10px;letter-spacing:2px;color:#555;margin-bottom:4px;margin-top:12px'>JUSTIFICATION</p>"
                        f"<p style='font-size:12px;color:#888;line-height:1.6'>{justif}</p>",
                        unsafe_allow_html=True,
                    )

                if surroundings:
                    st.markdown(
                        f"<p style='font-size:10px;letter-spacing:2px;color:#555;margin-bottom:4px;margin-top:12px'>SCENE</p>"
                        f"<p style='font-size:11px;color:#555;line-height:1.6'>{surroundings}</p>",
                        unsafe_allow_html=True,
                    )

                # similarity bar
                st.markdown(
                    f"<div style='margin-top:12px'>"
                    f"<div style='font-size:10px;color:#555;letter-spacing:2px;margin-bottom:6px'>SIMILARITY</div>"
                    f"<div style='height:3px;background:#222'>"
                    f"<div style='width:{pct}%;height:100%;background:#c8ff00'></div>"
                    f"</div>"
                    f"<div style='font-size:11px;color:#c8ff00;margin-top:4px'>{score:.3f}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # ── JSON export ────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="EXPORT JSON",
            data=json.dumps({"advice": advice, "retrieved": [r["meta"] for r in retrieved]}, indent=2),
            file_name="advice.json",
            mime="application/json",
        )