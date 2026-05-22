"""
PCB Defect Detection — FastAPI Backend
Model : YOLOv10m (NMS-free) via ONNX Runtime
Input : 640×640 letterboxed, float32, normalised [0,1]
Output: raw tensor [1, N, 6] → [x1,y1,x2,y2, score, class_id]
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import Column, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH  = Path("models/best.onnx")
INPUT_SIZE  = 640
CONF_THRESH = 0.45
DB_URL      = "sqlite:///./inspections.db"

CLASSES = [
    "falsecopper", "missinghole", "mousebite", "opencircuit",
    "pinhole",     "scratch",     "shortcircuit", "spur",
]

# ── QA Rule Engine ────────────────────────────────────────────────────────────

DEFECT_RULES = {
    "mousebite":    {"severity": "Medium",   "cause": "Possible over-etching instability",  "action": "Inspect etching machine calibration"},
    "shortcircuit": {"severity": "Critical", "cause": "Copper bridging during fabrication",  "action": "Immediate production halt and manual inspection"},
    "missinghole":  {"severity": "High",     "cause": "Drilling alignment issue",            "action": "Recalibrate CNC drilling equipment"},
    "opencircuit":  {"severity": "Critical", "cause": "Broken conductive trace",             "action": "Inspect trace manufacturing process"},
    "pinhole":      {"severity": "Low",      "cause": "Minor copper inconsistency",          "action": "Schedule maintenance inspection"},
    "scratch":      {"severity": "Low",      "cause": "Mechanical surface damage",           "action": "Inspect handling process"},
    "falsecopper":  {"severity": "Medium",   "cause": "Copper residue during etching",       "action": "Inspect etching process"},
    "spur":         {"severity": "Medium",   "cause": "Excess conductive material",          "action": "Check fabrication calibration"},
}
SEVERITY_RANK = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0}

# ── Database ───────────────────────────────────────────────────────────────────

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})

class Base(DeclarativeBase):
    pass

class InspectionRecord(Base):
    __tablename__ = "inspections"

    id             = Column(Integer, primary_key=True, index=True)
    timestamp      = Column(String,  nullable=False)   # ISO-8601 UTC
    status         = Column(String,  nullable=False)   # "PASS" | "FAIL"
    latency_ms     = Column(Float,   nullable=False)
    defect_summary = Column(String,  nullable=False)   # JSON string

Base.metadata.create_all(bind=engine)

# ── App & model initialisation ─────────────────────────────────────────────────

app = FastAPI(title="PCB Defect Inspector", version="2.0")

session: ort.InferenceSession = None

@app.on_event("startup")
def load_model():
    global session
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH.resolve()}")
    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )
    print(f"[startup] Model loaded: {MODEL_PATH}  |  "
          f"Input  : {session.get_inputs()[0].shape}  |  "
          f"Output : {session.get_outputs()[0].shape}")


# ── Preprocessing ──────────────────────────────────────────────────────────────

def letterbox(img_bgr: np.ndarray, target: int = INPUT_SIZE):
    """
    Resize to target×target preserving aspect ratio via grey padding.
    Returns padded image plus scale and (dw, dh) offsets for inverse transform.
    """
    h, w   = img_bgr.shape[:2]
    scale  = min(target / h, target / w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas  = np.full((target, target, 3), 114, dtype=np.uint8)
    dw, dh  = (target - new_w) // 2, (target - new_h) // 2
    canvas[dh:dh + new_h, dw:dw + new_w] = resized

    return canvas, scale, dw, dh


def preprocess(img_bgr: np.ndarray):
    """BGR → RGB → letterbox → float32 [0,1] → NCHW batch tensor."""
    img_rgb          = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lb_img, scale, dw, dh = letterbox(img_rgb)

    tensor = lb_img.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, 0)

    return tensor, scale, dw, dh


# ── Post-processing ────────────────────────────────────────────────────────────

def postprocess(raw: np.ndarray, scale: float, dw: int, dh: int,
                orig_h: int, orig_w: int) -> list[dict]:
    """
    Parse YOLOv10m NMS-free output [1, N, 6] → [x1,y1,x2,y2,conf,cls_id].
    Maps letterbox coordinates back to original image pixel space.
    """
    detections = []

    for pred in raw[0]:
        x1, y1, x2, y2, conf, cls_id = pred

        if conf < CONF_THRESH:
            continue

        x1 = float(max(0, min((x1 - dw) / scale, orig_w)))
        y1 = float(max(0, min((y1 - dh) / scale, orig_h)))
        x2 = float(max(0, min((x2 - dw) / scale, orig_w)))
        y2 = float(max(0, min((y2 - dh) / scale, orig_h)))

        detections.append({
            "defect_type": CLASSES[int(cls_id)],
            "confidence":  round(float(conf), 4),
            "bbox":        [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


# ── Persistence helper ─────────────────────────────────────────────────────────

def save_record(status: str, latency_ms: float, detections: list[dict]) -> None:
    """Write one inspection result to SQLite synchronously."""
    record = InspectionRecord(
        timestamp      = datetime.now(timezone.utc).isoformat(),
        status         = status,
        latency_ms     = round(latency_ms, 2),
        defect_summary = json.dumps(detections),
    )
    with Session(engine) as db:
        db.add(record)
        db.commit()


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/inspect")
async def inspect(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    arr       = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr   = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    orig_h, orig_w = img_bgr.shape[:2]

    t0                    = time.perf_counter()
    tensor, scale, dw, dh = preprocess(img_bgr)
    input_name            = session.get_inputs()[0].name
    raw_output            = session.run(None, {input_name: tensor})[0]
    latency_ms            = (time.perf_counter() - t0) * 1000

    detections = postprocess(raw_output, scale, dw, dh, orig_h, orig_w)
    status     = "FAIL" if detections else "PASS"

    save_record(status, latency_ms, detections)

    return JSONResponse({
        "status":        status,
        "latency_ms":    round(latency_ms, 2),
        "defects_found": len(detections),
        "details":       detections,
    })


@app.post("/generate_report")
async def generate_report(payload: dict):
    """Deterministic rule-based QA report — no external dependencies."""
    if payload.get("status") == "PASS":
        return {"report": "Status: PASS\nNo action required."}

    defects = payload.get("details", [])

    # Aggregate counts per defect type
    counts: dict[str, int] = {}
    for d in defects:
        counts[d["defect_type"]] = counts.get(d["defect_type"], 0) + 1

    # Determine highest-severity defect
    top_type = max(
        counts,
        key=lambda t: SEVERITY_RANK.get(DEFECT_RULES.get(t, {}).get("severity", "Low"), 0),
    )
    rule      = DEFECT_RULES.get(top_type, {"severity": "Unknown", "cause": "Unknown", "action": "Manual review required"})
    detected  = ", ".join(f"{t} x{n}" for t, n in counts.items())

    report = (
        f"Status            : FAIL\n"
        f"Detected          : {detected}\n"
        f"Severity          : {rule['severity']}\n"
        f"Likely Cause      : {rule['cause']}\n"
        f"Recommended Action: {rule['action']}"
    )
    return {"report": report}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": session is not None}
