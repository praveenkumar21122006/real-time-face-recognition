import os

# Camera sources: 0 = default webcam, or RTSP URL / video file
CAMERA_SOURCES = {
    "main_entrance": 0,
    # "side_entrance": "rtsp://admin:pass@192.168.1.64/stream",
    # "lobby": "videos/lobby.mp4"
}

# Recognition tuning
TOLERANCE = 0.42          # lower = stricter (0.35-0.50 for LBPH/embedding fallback)
FRAME_SKIP = 2            # process every Nth frame for performance
RESIZE_FACTOR = 0.5       # downscale for detection speed
FACE_SIZE = (128, 128)    # normalized face size for embedding

# Storage - use /tmp on Vercel (read-only filesystem)
_BASE = "/tmp/data" if os.getenv("VERCEL") else "data"
DB_PATH = f"{_BASE}/staff_db.json"
ENCODING_DIR = f"{_BASE}/encodings"
LOG_PATH = f"{_BASE}/entry_log.json"
THUMB_DIR = f"{_BASE}/thumbs"

# Access control
DOOR_UNLOCK_SECONDS = 3
RECOGNITION_COOLDOWN = 5  # seconds before same person triggers again

for p in [ENCODING_DIR, THUMB_DIR, _BASE]:
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass
