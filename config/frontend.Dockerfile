# ── Frontend — Streamlit client ───────────────────────────────────────────────
FROM python:3.10-slim

# Same OpenCV system libs required — cv2 is used for image decode and BGR→RGB.
RUN apt-get update && apt-get install -y --no-install-recommends \
                libgl1 \
                libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY config/requirements.txt .
RUN pip install --no-cache-dir \
        streamlit \
        requests \
        opencv-python \
        numpy

COPY frontend/app.py .

EXPOSE 8501

# --server.address=0.0.0.0 makes Streamlit reachable outside the container.
# --server.fileWatcherType=none disables hot-reload (not needed in production).
CMD ["python3", "-m", "streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.fileWatcherType=none"]
