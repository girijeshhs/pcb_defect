# ── Backend — FastAPI + ONNX Runtime ─────────────────────────────────────────
FROM python:3.10-slim

# libgl1 + libglib2 are required by opencv-python's libGL/libgthread dependency.
# Without these, cv2.imdecode raises "libGL.so.1: cannot open shared object file".
RUN apt-get update && apt-get install -y --no-install-recommends \
                libgl1 \
                libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install deps before copying source — maximises Docker layer cache reuse.
# If requirements.txt doesn't change, this layer is never rebuilt.
COPY requirements.txt .
RUN pip install --no-cache-dir \
        fastapi \
        uvicorn \
        python-multipart \
        onnxruntime \
        opencv-python \
        numpy

# Copy only the backend source — the model directory is mounted as a volume,
# so best.onnx is never baked into the image (keeps the image size small).
COPY main.py .

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
