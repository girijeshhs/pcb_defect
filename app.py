"""
PCB Defect Detection — Streamlit Client
FastAPI backend: reads from API_URL env var (default: http://127.0.0.1:8000/inspect)
"""

import os

import cv2
import numpy as np
import requests
import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────────────

# Inside Docker, Compose sets API_URL=http://api:8000/inspect so the client
# reaches the backend via Docker's internal DNS (service name = hostname).
# Locally, the fallback points to localhost for direct uvicorn runs.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/inspect")

# BGR tuples (not RGB) — cv2 drawing functions operate in BGR colour space.
# Colours chosen for contrast against the dark green PCB solder-mask layer.
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

def decode_image(uploaded_file) -> np.ndarray:
    """
    Streamlit gives a memory buffer, not a file path.
    frombuffer + imdecode replicates cv2.imread() from raw bytes.
    imdecode always returns BGR — we convert to RGB only at display time.
    """
    raw = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — file may be corrupt.")
    return img


def draw_defects(img_bgr: np.ndarray, defects: list) -> np.ndarray:
    """
    Draws colour-coded bounding boxes + label backgrounds onto a BGR image.
    Coordinates from the API are already in the original image pixel space,
    so no scaling is needed. We clamp to image bounds to prevent cv2 errors.
    """
    h, w = img_bgr.shape[:2]

    for d in defects:
        label   = f"{d['defect_type']}  {d['confidence']:.0%}"
        color   = DEFECT_COLORS.get(d["defect_type"], DEFAULT_COLOR)
        bbox    = d["bbox"]

        # Clamp float coords to valid integer pixel indices
        x1 = int(max(0, min(bbox[0], w - 1)))
        y1 = int(max(0, min(bbox[1], h - 1)))
        x2 = int(max(0, min(bbox[2], w - 1)))
        y2 = int(max(0, min(bbox[3], h - 1)))

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, THICKNESS)

        # Measure text to build a perfectly-fitting background rect
        (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        lbl_h = th + baseline + 2 * PAD

        # Place label above box; fall back to inside-top if near image edge
        if y1 - lbl_h - 6 >= 0:
            ly1, ly2 = y1 - lbl_h - 6, y1 - 6
        else:
            ly1, ly2 = y1 + 6, y1 + lbl_h + 6

        lx2 = min(x1 + tw + 2 * PAD, w - 1)

        # Draw filled background rect first, then text on top
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
        hex_c = f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"   # BGR → RGB hex
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
            f'<div style="width:13px;height:13px;border-radius:3px;background:{hex_c}"></div>'
            f'<span style="font-size:.83rem;color:#ccc">{cls}</span></div>',
            unsafe_allow_html=True)
    st.divider()
    st.caption(f"API → `{API_URL}`")

# ── Hero header & uploader ─────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <h1>🔬 PCB Defect Inspector</h1>
  <p>8-class YOLOv10m · Real-time OpenCV visualisation · FastAPI backend</p>
</div>""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload a PCB image", type=["jpg","jpeg","png","bmp","tiff"])

# ── Main flow ──────────────────────────────────────────────────────────────────

if uploaded:
    try:
        img_bgr = decode_image(uploaded)
    except ValueError as e:
        st.error(str(e)); st.stop()

    with st.spinner("Inspecting…"):
        try:
            result = call_api(img_bgr, uploaded.name)
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot reach API at `{API_URL}`. Is the server running?"); st.stop()
        except requests.exceptions.HTTPError as e:
            st.error(f"API error: {e}"); st.stop()

    status    = result.get("status", "UNKNOWN")
    latency   = result.get("latency_ms", 0.0)
    defects   = result.get("details", [])
    n_defects = result.get("defects_found", len(defects))
    img_h, img_w = img_bgr.shape[:2]

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status",      status)
    c2.metric("Latency",     f"{latency:.1f} ms")
    c3.metric("Defects",     n_defects)
    c4.metric("Resolution",  f"{img_w}×{img_h}")

    # Draw annotations on a copy, then convert BGR→RGB for Streamlit display.
    # cv2 draws in BGR; st.image() expects RGB — one conversion at the end.
    annotated_rgb = cv2.cvtColor(draw_defects(img_bgr.copy(), defects),
                                 cv2.COLOR_BGR2RGB)
    original_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    st.divider()
    col_l, col_r = st.columns(2, gap="medium")
    col_l.markdown("**📷 Original**")
    col_l.image(original_rgb,  use_container_width=True)
    col_r.markdown("**🔬 Annotated**")
    col_r.image(annotated_rgb, use_container_width=True)

    # Defect table
    if defects:
        st.divider()
        st.markdown("**📋 Detections**")
        for i, d in enumerate(defects, 1):
            bgr     = DEFECT_COLORS.get(d["defect_type"], DEFAULT_COLOR)
            hex_c   = f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"
            b       = d["bbox"]
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