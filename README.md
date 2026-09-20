# Real-Time Face Recognition Entry System

Scans camera feeds at building entrances to match staff credentials instantly (<100ms on CPU).

## Features
- **Multi-entrance** RTSP / webcam / video file support (`config.CAMERA_SOURCES`)
- **Instant matching**: frame-skip + downscale + HOG/Haar detection, 128-d embedding, cosine/Euclidean threshold
- **Hybrid engine**: uses `face_recognition` (dlib) if installed, otherwise pure OpenCV fallback (no dlib compile needed)
- **Enroll in UI**: 1-5 photos per staff -> embeddings stored in `data/staff_db.json` + `.npy`
- **Access log** with cooldown (no spam), door unlock simulation
- **REST + MJPEG stream** via FastAPI

## Quick Start
```bash
pip install -r requirements.txt
# optional for best accuracy:
pip install dlib face_recognition

python main.py
# open http://localhost:8000
```

## Config `config.py`
```python
CAMERA_SOURCES = {"main_entrance": 0, "side": "rtsp://..."}
TOLERANCE = 0.42
FRAME_SKIP = 2
RECOGNITION_COOLDOWN = 5
```

## API
- `POST /api/staff/enroll` form: `staff_id, name, role, files[]` -> enroll
- `GET /api/staff` -> list
- `DELETE /api/staff/{id}`
- `POST /api/recognize` file -> instant single-image check
- `GET /video_feed/{entrance}` -> MJPEG
- `GET /api/log?n=50`
- `GET /health`

## How it works
1. `CameraStream` thread reads frames, every Nth frame runs `detect_faces` -> `encode_face`
2. `StaffDB.match()` compares query embedding to all staff embeddings (L2 or cosine)
3. If distance < `TOLERANCE` -> `GRANTED` + log, else `DENIED`
4. Frontend polls log and shows live overlay (green/red boxes)

## Production notes
- For GPU/large scale: swap fallback with InsightFace / FaceNet ONNX
- Add liveness (blink/ depth) to prevent photo spoof
- Put `uvicorn --workers 1` (camera threads are not fork-safe) behind nginx
- Secure enroll endpoint with auth
