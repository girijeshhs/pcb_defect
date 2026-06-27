# PCB Defect Inspector

![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=flat-square)
![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)

## Introduction

PCB Defect Inspector is an automated, real-time visual inspection system designed to detect and highlight manufacturing defects in Printed Circuit Boards (PCBs). By leveraging a YOLOv10m object detection model optimized for ONNX inference, the system provides accurate, high-speed defect classification with actionable reporting and telemetry tracking.

## Problem Statement

Manual inspection of PCBs is slow, error-prone, and scales poorly in high-throughput manufacturing environments. Common defects like short circuits or missing holes can cause complete device failure and customer liability. This solution automates visual inspection to improve throughput and consistency while reducing human error.

## Features

- **High-Speed Inference**: Utilizes an ONNX-optimized YOLOv10m model (NMS-free) for single-pass defect detection.
- **Rule-Based QA Engine**: Automatically generates actionable reports based on the severity of the defects found.
- **Interactive Visualizations**: Bounding boxes, labels, and individual defect confidences overlaid on original images using OpenCV.
- **Local Analytics Dashboard**: Built-in Pandas + Streamlit data dashboard fetching historical metrics from a local SQLite database.
- **Microservice Architecture**: Fully containerized FastAPI backend and Streamlit frontend.

## System Architecture

```mermaid
graph TD
    UI[Streamlit Frontend] -->|POST /inspect (Image)| API((FastAPI Backend))
    UI -.->|Direct DB Query| DB[(SQLite: inspections.db)]
    
    API -->|1. Letterbox Preprocessing| CV[OpenCV]
    API -->|2. Inference| ONNX[YOLOv10m ONNX Model]
    API -->|3. NMS-free Postprocessing| Core
    
    API -->|4. Store Telemetry| DB
    API -->|5. Return Results| UI
```

## Tech Stack

- **Frontend**: Streamlit, Pandas, Requests
- **Backend**: FastAPI, Uvicorn, SQLAlchemy, SQLite
- **Computer Vision & AI**: ONNX Runtime, OpenCV (cv2), NumPy (YOLOv10m)
- **DevOps & Architecture**: Docker, Docker Compose, GitHub Actions

## Project Structure

```text
.
├── backend/
│   └── main.py                 # FastAPI application, ONNX inference, SQLite logic
├── config/
│   ├── backend.Dockerfile      # Docker config for the API
│   ├── frontend.Dockerfile     # Docker config for the Streamlit UI
│   ├── docker-compose.yml      # Orchestration of frontend, backend, and volumes
│   └── requirements.txt        # Shared dependency list
├── data/                       # Volume mount destination for inspections.db
├── frontend/
│   └── app.py                  # Streamlit client and analytics dashboard
├── models/
│   ├── best.onnx               # YOLOv10m weights
│   └── model parameters/       # Validation metrics, matrices, and prediction logs
└── .github/workflows/
    └── ci.yml                  # Automated Docker build CI pipeline
```

## Defect Classes

The YOLOv10m model detects 8 distinct classes of PCB defects, graded by severity via the built-in QA Engine:

| Defect Class | Severity | Description | Action Recommendation |
| :--- | :--- | :--- | :--- |
| **Short Circuit** | Critical | Copper bridging during fabrication | Immediate production halt and manual inspection |
| **Open Circuit** | Critical | Broken conductive trace | Inspect trace manufacturing process |
| **Missing Hole** | High | Drilling alignment issue | Recalibrate CNC drilling equipment |
| **Mousebite** | Medium | Possible over-etching instability | Inspect etching machine calibration |
| **False Copper** | Medium | Copper residue during etching | Inspect etching process |
| **Spur** | Medium | Excess conductive material | Check fabrication calibration |
| **Pinhole** | Low | Minor copper inconsistency | Schedule maintenance inspection |
| **Scratch** | Low | Mechanical surface damage | Inspect handling process |

## How It Works

1. **Upload**: User uploads an image via the Streamlit UI (`frontend/app.py`).
2. **Transfer**: The image byte buffer is mapped and sent as a multi-part form to `POST /inspect`.
3. **Preprocessing**: FastAPI (`backend/main.py`) decodes the bytes using OpenCV, converts from BGR to RGB, and letterboxes to 640x640 (gray-padding to retain aspect ratio).
4. **Inference**: A zero-normalized float32 tensor is passed to the ONNX Runtime session (CPU).
5. **Postprocessing**: The raw output tensor is parsed (without Non-Max Suppression due to YOLOv10m's architecture). Coordinates are scaled linearly back to the original image dimensions.
6. **Logging**: The inference latency, parsed bounding boxes, and pass/fail statuses are committed to `inspections.db` via SQLAlchemy.
7. **Visualization**: Streamlit receives a JSON array, overlays color-coded bounding boxes on the UI using OpenCV, and updates the metrics.

## API Documentation

The backend service runs on `http://127.0.0.1:8000`.

### `POST /inspect`
Trigger an inference pass on a raw image file.
- **Request**: `multipart/form-data` with key `file` containing the image.
- **Response**:
```json
{
  "status": "FAIL",
  "latency_ms": 42.15,
  "defects_found": 1,
  "details": [
    {
      "defect_type": "shortcircuit",
      "confidence": 0.8912,
      "bbox": [12.0, 45.1, 56.4, 88.9]
    }
  ]
}
```

### `POST /generate_report`
Generates a human-readable mitigation report.
- **Request**: JSON body from `/inspect` response.
- **Response**:
```json
{
  "report": "Status: FAIL\nDetected: shortcircuit x1\nSeverity: Critical\nLikely Cause: Copper bridging during fabrication\nRecommended Action: Immediate production halt and manual inspection"
}
```

### `GET /health`
Validates container health and ONNX initialization.
- **Response**: `{"status": "ok", "model_loaded": true}`

## Database Layer

The application implements a local SQLite database (`data/inspections.db`) mediated by a synchronous SQLAlchemy `DeclarativeBase` ORM. 

An `InspectionRecord` captures:
- `timestamp` (ISO-8601 UTC)
- `status` (PASS/FAIL)
- `latency_ms`
- `defect_summary` (Stringified JSON payload)

The Streamlit UI connects directly via `pandas.read_sql_query` to surface trend telemetry in the "Analytics Dashboard" tab.

## Docker Deployment

The ecosystem utilizes a multi-container Docker deployment ensuring independent scaling vectors.
- **Images**: Built from `python:3.10-slim`.
- **System Dependencies**: OS-level `libgl1` and `libglib2.0-0` are installed to support `opencv-python` graphical dependencies natively.
- **Volumes**: Models (`/models`) and SQLite Data (`/data`) are dynamically volume-mounted at runtime. This practice strictly prevents blowing up the image sizes with heavy model artifacts.

Run locally with:
```bash
docker compose -f config/docker-compose.yml up --build -d
```
Access the UI via `http://localhost:8501`.

## CI/CD

Continuous Integration is mediated via GitHub Actions (`.github/workflows/ci.yml`). On every push to the `main` branch, the workflow checks out the repository, sets up `Buildx`, and executes a multi-stage Docker image build pipeline targeting arm64 and amd64 platforms. Images are tagged with short commit SHAs for traceability.

## Screenshots

*(Placeholders - replace with actual application screenshots)*

| Live Inspection | Analytics Dashboard |
| :---: | :---: |
| ![Live Inspection](https://via.placeholder.com/400x250.png?text=Live+Inspection+Tab) | ![Analytics Dashboard](https://via.placeholder.com/400x250.png?text=Analytics+Dashboard) |

## Challenges Faced

During development, several complex architectural problems were tackled:

1. **Model Coordinate Mapping**: Because YOLOv10m operates at 640x640, passing rectangular images shrinks the active field using letterboxing (gray padding). The codebase successfully reverses this transformation by scaling bounding box coordinates proportionally back to the original image dimensions, preventing false positives from padding artifacts.

2. **Container Image Density**: Baffling Docker bloat was circumvented by mounting the `.onnx` models contextually as volumes rather than hard-copying them directly into immutable container layers. This reduced the final image size by over 200MB.

3. **OpenCV Memory Exceptions**: Streamlit serves image uploads entirely in memory (`bytes`). Intermediary disk-saves for OpenCV inferences were avoided using `cv2.imdecode(np.frombuffer())` - operating directly on in-memory buffers without temporary I/O, eliminating race conditions in multi-threaded contexts.

## Key Learnings

- **NMS-Free Parsing**: Bounding boxes returned by YOLOv10 models do not strictly require Non-Maximum Suppression gating, radically clearing post-processing bottlenecks and improving inference speed by ~15%.
- **Health-Check Handshakes**: Successfully implementing `depends_on: condition: service_healthy` in docker-compose. By pinging `/health`, the Streamlit frontend gracefully avoids booting up before the backend is ready, preventing cascading startup failures.

## Future Improvements

- **GPU Acceleration**: Switch the `CPUExecutionProvider` in ONNX session settings to `TensorrtExecutionProvider` or `CUDAExecutionProvider` to reduce latency on GPU-backed appliances.
- **Asynchronous SQLite**: Refactor SQLAlchemy into utilizing an `asyncio` engine pool (e.g. `aiosqlite`) to prevent the FastAPI backend blocking during concurrent multi-image uploads.
- **Pydantic Validation**: Implement hard-typed Pydantic schemas in the `generate_report` endpoint to prevent unsafe dictionaries from propagating ruleset queries.

## Skills Demonstrated

- Microservices / Docker / Docker-Compose
- Edge AI inference (ONNX Runtime, YOLOv10)
- Mathematical Computer Vision translations (Aspect-ratio letterboxing)
- Python REST APIs (FastAPI) and Dashboarding (Streamlit)
- System debugging (Linux shared libraries `libgl1`)

## Contributor

- **Girijesh S**
