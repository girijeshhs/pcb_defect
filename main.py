"""
PCB Defect Detection — FastAPI Backend
Model : YOLOv10m (NMS-free) via ONNX Runtime
Input : 640×640 letterboxed, float32, normalised [0,1]
Output: raw tensor [1, N, 6] → [x1,y1,x2,y2, score, class_id]
"""

import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH  = Path("models/best.onnx")
INPUT_SIZE  = 640          # model expects 640×640
CONF_THRESH = 0.45         # minimum confidence to report a defect

CLASSES = [
    "falsecopper", "missinghole", "mousebite", "opencircuit",
    "pinhole",     "scratch",     "shortcircuit", "spur",
]

# ── App & model initialisation ─────────────────────────────────────────────────

app = FastAPI(title="PCB Defect Inspector", version="1.0")

# Load ONNX session once at startup — heavy operation, must not repeat per request
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
    Resize image to target×target while preserving aspect ratio by padding.
    Returns the padded image plus the scale and (dw, dh) offsets needed to
    invert the transform and recover original-image coordinates.
    """
    h, w = img_bgr.shape[:2]
    scale = min(target / h, target / w)           # uniform scale factor
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Padding to reach exactly target×target (grey: 114 matches YOLO convention)
    canvas = np.full((target, target, 3), 114, dtype=np.uint8)
    dw, dh = (target - new_w) // 2, (target - new_h) // 2
    canvas[dh:dh + new_h, dw:dw + new_w] = resized

    return canvas, scale, dw, dh


def preprocess(img_bgr: np.ndarray):
    """
    BGR → RGB → letterbox → float32 [0,1] → NCHW batch tensor.
    OpenCV reads in BGR; YOLO was trained on RGB — channel swap is mandatory.
    """
    img_rgb         = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lb_img, scale, dw, dh = letterbox(img_rgb)

    tensor = lb_img.astype(np.float32) / 255.0   # normalise [0,1]
    tensor = np.transpose(tensor, (2, 0, 1))      # HWC → CHW
    tensor = np.expand_dims(tensor, 0)            # CHW → NCHW (batch=1)

    return tensor, scale, dw, dh


# ── Post-processing ────────────────────────────────────────────────────────────

def postprocess(raw: np.ndarray, scale: float, dw: int, dh: int,
                orig_h: int, orig_w: int) -> list[dict]:
    """
    Parse YOLOv10m NMS-free output tensor.

    YOLOv10 does NOT require NMS — it outputs at most N deduplicated predictions.
    Tensor shape: [1, N, 6]  →  each row: [x1, y1, x2, y2, confidence, class_id]
    All bbox coordinates are in letterboxed 640×640 pixel space and must be
    mapped back to the original image dimensions using the inverse letterbox transform.
    """
    predictions = raw[0]           # shape: [N, 6]  (drop batch dim)
    detections  = []

    for pred in predictions:
        x1, y1, x2, y2, conf, cls_id = pred

        if conf < CONF_THRESH:
            continue

        # Inverse letterbox: remove padding offset, then undo scale
        # This maps from 640×640 letterbox space → original image pixel space
        x1 = (x1 - dw) / scale
        y1 = (y1 - dh) / scale
        x2 = (x2 - dw) / scale
        y2 = (y2 - dh) / scale

        # Clamp to original image bounds
        x1 = float(max(0, min(x1, orig_w)))
        y1 = float(max(0, min(y1, orig_h)))
        x2 = float(max(0, min(x2, orig_w)))
        y2 = float(max(0, min(y2, orig_h)))

        detections.append({
            "defect_type": CLASSES[int(cls_id)],
            "confidence":  round(float(conf), 4),
            "bbox":        [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        })

    # Sort by confidence descending for cleaner JSON output
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


# ── Endpoint ───────────────────────────────────────────────────────────────────

@app.post("/inspect")
async def inspect(file: UploadFile = File(...)):
    # Decode uploaded bytes → OpenCV BGR array
    raw_bytes = await file.read()
    arr       = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr   = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    orig_h, orig_w = img_bgr.shape[:2]

    # Preprocess → infer → postprocess
    t0               = time.perf_counter()
    tensor, scale, dw, dh = preprocess(img_bgr)
    input_name       = session.get_inputs()[0].name
    raw_output       = session.run(None, {input_name: tensor})[0]
    latency_ms       = (time.perf_counter() - t0) * 1000

    detections = postprocess(raw_output, scale, dw, dh, orig_h, orig_w)

    return JSONResponse({
        "status":        "FAIL" if detections else "PASS",
        "latency_ms":    round(latency_ms, 2),
        "defects_found": len(detections),
        "details":       detections,
    })


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": session is not None}
