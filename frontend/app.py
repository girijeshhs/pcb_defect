"""
PCB Defect Detection — Streamlit Client
FastAPI backend: reads from API_URL env var (default: http://127.0.0.1:8000/inspect)
Analytics tab: reads directly from inspections.db via sqlite3 + pandas
"""

import hashlib
import os
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────────────

API_URL     = os.getenv("API_URL",    "http://127.0.0.1:8000/inspect")
DB_PATH     = Path(os.getenv("DB_PATH", str(Path(__file__).resolve().parent.parent / "inspections.db")))

DEFECT_COLORS = {
    "falsecopper":  (0,   200, 255),
    "missinghole":  (255, 50,  50 ),
    "mousebite":    (50,  255, 200),
    "opencircuit":  (0,   80,  255),
    "pinhole":      (230, 0,   200),
    "scratch":      (50,  220, 255),
    "shortcircuit": (255, 130, 0  ),
    "spur":         (0,   255, 130),
}
DEFAULT_COLOR = (255, 255, 255)

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS  = 2
PAD        = 4

# ── Helper functions ───────────────────────────────────────────────────────────

def decode_image(raw_bytes: bytes) -> np.ndarray:
    """
    Streamlit gives a memory buffer, not a file path.
    frombuffer + imdecode replicates cv2.imread() from raw bytes.
    """
    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — file may be corrupt.")
    return img


def draw_defects(img_bgr: np.ndarray, defects: list) -> np.ndarray:
    """
    Colour-coded bounding boxes + label backgrounds on a BGR image.
    Coordinates are already in original pixel space — no scaling needed.
    """
    h, w = img_bgr.shape[:2]

    for d in defects:
        label = f"{d['defect_type']}  {d['confidence']:.0%}"
        color = DEFECT_COLORS.get(d["defect_type"], DEFAULT_COLOR)
        bbox  = d["bbox"]

        x1 = int(max(0, min(bbox[0], w - 1)))
        y1 = int(max(0, min(bbox[1], h - 1)))
        x2 = int(max(0, min(bbox[2], w - 1)))
        y2 = int(max(0, min(bbox[3], h - 1)))

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, THICKNESS)

        (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        lbl_h = th + baseline + 2 * PAD

        if y1 - lbl_h - 6 >= 0:
            ly1, ly2 = y1 - lbl_h - 6, y1 - 6
        else:
            ly1, ly2 = y1 + 6, y1 + lbl_h + 6

        lx2 = min(x1 + tw + 2 * PAD, w - 1)

        cv2.rectangle(img_bgr, (x1, ly1), (lx2, ly2), color, -1)
        cv2.putText(img_bgr, label,
                    (x1 + PAD, ly2 - baseline - PAD),
                    FONT, FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA)

    return img_bgr


def call_api(img_bgr: np.ndarray, filename: str) -> dict:
    _, buf = cv2.imencode(".jpg", img_bgr)
    resp = requests.post(API_URL,
                         files={"file": (filename, buf.tobytes(), "image/jpeg")},
                         timeout=30)
    resp.raise_for_status()
    return resp.json()


def load_inspection_history() -> pd.DataFrame:
    """Read all inspection records from SQLite and return a tidy DataFrame."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query(
            "SELECT id, timestamp, status, latency_ms, defect_summary FROM inspections ORDER BY id",
            con,
        )
    if df.empty:
        return df

    df["timestamp"]    = pd.to_datetime(df["timestamp"], utc=True)
    df["defect_count"] = df["defect_summary"].apply(
        lambda s: len(__import__("json").loads(s)) if s else 0
    )
    return df


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="PCB Defect Inspector", page_icon="🔬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.hero { background: linear-gradient(135deg,#0f0c29,#302b63,#24243e);
        border-radius:14px; padding:1.8rem 2.4rem; margin-bottom:1.2rem; }
.hero h1 { color:#f0f0f0; font-size:2rem; margin:0; }
.hero p  { color:#a0a8c0; margin:.3rem 0 0; }
[data-testid="stMetric"] { background:#1a1a2e; border:1px solid #2d2d50;
                            border-radius:12px; padding:.9rem 1.1rem; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🎨 Defect Legend")
    for cls, bgr in DEFECT_COLORS.items():
        hex_c = f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
            f'<div style="width:13px;height:13px;border-radius:3px;background:{hex_c}"></div>'
            f'<span style="font-size:.83rem;color:#ccc">{cls}</span></div>',
            unsafe_allow_html=True)
    st.divider()
    st.caption(f"API → `{API_URL}`")

# ── Hero header ────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <h1>🔬 PCB Defect Inspector</h1>
  <p>8-class YOLOv10m · Real-time OpenCV visualisation · FastAPI + SQLite backend</p>
</div>""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_inspect, tab_analytics = st.tabs(["🔬 Live Inspection", "📊 Analytics Dashboard"])

# ══ Tab 1: Live Inspection ═════════════════════════════════════════════════════

with tab_inspect:
    uploaded = st.file_uploader("Upload a PCB image", type=["jpg","jpeg","png","bmp","tiff"])

    if uploaded:
        raw_bytes = uploaded.getvalue()
        upload_sig = hashlib.md5(raw_bytes).hexdigest()
        try:
            img_bgr = decode_image(raw_bytes)
        except ValueError as e:
            st.error(str(e)); st.stop()

        if st.session_state.get("last_upload_sig") != upload_sig:
            with st.spinner("Inspecting…"):
                try:
                    result = call_api(img_bgr, uploaded.name)
                except requests.exceptions.ConnectionError:
                    st.error(f"Cannot reach API at `{API_URL}`. Is the server running?"); st.stop()
                except requests.exceptions.HTTPError as e:
                    st.error(f"API error: {e}"); st.stop()
            st.session_state["last_upload_sig"] = upload_sig
            st.session_state["last_result"] = result
        else:
            result = st.session_state.get("last_result", {})

        status    = result.get("status", "UNKNOWN")
        latency   = result.get("latency_ms", 0.0)
        defects   = result.get("details", [])
        n_defects = result.get("defects_found", len(defects))
        img_h, img_w = img_bgr.shape[:2]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Status",     status)
        c2.metric("Latency",    f"{latency:.1f} ms")
        c3.metric("Defects",    n_defects)
        c4.metric("Resolution", f"{img_w}×{img_h}")

        annotated_rgb = cv2.cvtColor(draw_defects(img_bgr.copy(), defects), cv2.COLOR_BGR2RGB)
        original_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        st.divider()
        col_l, col_r = st.columns(2, gap="medium")
        col_l.markdown("**📷 Original**")
        col_l.image(original_rgb,  use_container_width=True)
        col_r.markdown("**🔬 Annotated**")
        col_r.image(annotated_rgb, use_container_width=True)

        if defects:
            st.divider()
            st.markdown("**📋 Detections**")
            for i, d in enumerate(defects, 1):
                bgr   = DEFECT_COLORS.get(d["defect_type"], DEFAULT_COLOR)
                hex_c = f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"
                b     = d["bbox"]
                st.markdown(
                    f'`{i}` &nbsp; '
                    f'<span style="background:{hex_c};color:#000;padding:1px 7px;'
                    f'border-radius:4px;font-size:.82rem">{d["defect_type"]}</span> &nbsp; '
                    f'**{d["confidence"]:.1%}** &nbsp; '
                    f'<span style="color:#888;font-size:.82rem">'
                    f'({b[0]:.0f},{b[1]:.0f})→({b[2]:.0f},{b[3]:.0f})</span>',
                    unsafe_allow_html=True)
        else:
            st.success("✅ No defects — board passed inspection.")

        with st.expander("Raw JSON"):
            st.json(result)

    else:
        st.info("Upload a PCB image above to start inspection.", icon="ℹ️")

# ══ Tab 2: Analytics Dashboard ═════════════════════════════════════════════════

with tab_analytics:
    st.markdown("#### Inspection History")

    if st.button("🔄 Refresh", key="refresh_analytics"):
        st.rerun()

    df = load_inspection_history()

    if df.empty:
        st.info("No inspection records yet. Run some inspections first.", icon="📭")
    else:
        st.dataframe(
            df[["id", "timestamp", "status", "latency_ms", "defect_count"]]
              .sort_values("id", ascending=False),
            use_container_width=True,
            hide_index=True,
        )