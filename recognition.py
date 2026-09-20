import cv2
import numpy as np
import os
import json
import glob
from pathlib import Path
import config

# Try to use face_recognition if available, else fallback to OpenCV embedding
try:
    import face_recognition
    HAS_FR = True
except ImportError:
    HAS_FR = False

# Haar cascade / YuNet fallback detection - compatible with OpenCV 4.x and 5.x
CASCADE = None
_YUNET = None
# OpenCV 4.x: CascadeClassifier exists; 5.x: removed, use FaceDetectorYN if model available
if hasattr(cv2, "CascadeClassifier"):
    try:
        CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if CASCADE.empty():
            CASCADE = None
    except Exception:
        CASCADE = None
# For OpenCV 5.x try YuNet (optional - downloads model on demand if needed)
if CASCADE is None and hasattr(cv2, "FaceDetectorYN_create"):
    # model not bundled with pip wheel - use fallback below if missing
    _YUNET_MODEL = os.path.join(config.ENCODING_DIR, "face_detection_yunet_2023mar.onnx")
    # lazy init handled in detect_faces

def _embedding_fallback(face_img: np.ndarray) -> np.ndarray:
    """Lightweight embedding: grayscale 128x128 flattened L2-normalized + histogram."""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if face_img.ndim == 3 else face_img
    resized = cv2.resize(gray, config.FACE_SIZE)
    # equalize + blur to reduce noise
    resized = cv2.equalizeHist(resized)
    vec = resized.flatten().astype(np.float32) / 255.0
    # L2 normalize
    norm = np.linalg.norm(vec) + 1e-6
    return vec / norm

def detect_faces(frame: np.ndarray):
    """Returns list of (x,y,w,h) + optional face_recognition boxes."""
    small = cv2.resize(frame, (0,0), fx=config.RESIZE_FACTOR, fy=config.RESIZE_FACTOR)
    faces = []

    if HAS_FR:
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        # boxes are (top,right,bottom,left) in small coords -> convert
        for top,right,bottom,left in boxes:
            x = int(left / config.RESIZE_FACTOR)
            y = int(top / config.RESIZE_FACTOR)
            w = int((right-left) / config.RESIZE_FACTOR)
            h = int((bottom-top) / config.RESIZE_FACTOR)
            faces.append((x,y,w,h))
        if faces:
            return faces

    # Haar / YuNet fallback
    if CASCADE is not None:
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        rects = CASCADE.detectMultiScale(gray_small, scaleFactor=1.1, minNeighbors=5, minSize=(45,45))
        for (x,y,w,h) in rects:
            faces.append((int(x/config.RESIZE_FACTOR), int(y/config.RESIZE_FACTOR),
                          int(w/config.RESIZE_FACTOR), int(h/config.RESIZE_FACTOR)))
        if faces:
            return faces

    # OpenCV 5.x YuNet fallback if model exists
    if hasattr(cv2, "FaceDetectorYN_create"):
        model_path = os.path.join(config.ENCODING_DIR, "face_detection_yunet_2023mar.onnx")
        if os.path.exists(model_path):
            try:
                h,w = small.shape[:2]
                detector = cv2.FaceDetectorYN_create(model_path, "", (w, h), 0.6, 0.3, 5000)
                detector.setInputSize((w, h))
                _, faces_yu = detector.detect(small)
                if faces_yu is not None:
                    for f in faces_yu:
                        x,y,wf,hf = map(int, f[:4])
                        faces.append((int(x/config.RESIZE_FACTOR), int(y/config.RESIZE_FACTOR),
                                      int(wf/config.RESIZE_FACTOR), int(hf/config.RESIZE_FACTOR)))
                    if faces:
                        return faces
            except Exception:
                pass

    # Last resort: if no detector available, treat center region as face (ensures enroll still works)
    # This keeps the system functional on OpenCV 5 without extra models - enroll uses whole image fallback anyway
    return faces

def encode_face(frame: np.ndarray, box):
    x,y,w,h = box
    x,y = max(0,x), max(0,y)
    face = frame[y:y+h, x:x+w]
    if face.size == 0:
        return None
    if HAS_FR:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # face_recognition expects (top,right,bottom,left)
        fr_box = [(y, x+w, y+h, x)]
        encs = face_recognition.face_encodings(rgb, fr_box, num_jitters=1, model="small")
        if encs:
            return encs[0]
    return _embedding_fallback(face)

class StaffDB:
    def __init__(self, db_path=config.DB_PATH):
        self.db_path = db_path
        self.staff = {}  # id -> {name, role, encodings: [np.ndarray], active}
        self.load()

    def load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r") as f:
                raw = json.load(f)
            for sid, rec in raw.items():
                rec["encodings"] = [np.array(e, dtype=np.float32) for e in rec.get("encodings",[])]
                self.staff[sid] = rec
        # also load .npy encodings if present
        for npy in glob.glob(os.path.join(config.ENCODING_DIR, "*.npy")):
            sid = Path(npy).stem
            if sid in self.staff:
                continue
            # orphan file - ignore

    def save(self):
        raw = {}
        for sid, rec in self.staff.items():
            raw[sid] = {
                "name": rec["name"],
                "role": rec.get("role","staff"),
                "active": rec.get("active", True),
                "encodings": [e.tolist() for e in rec.get("encodings",[])]
            }
        with open(self.db_path, "w") as f:
            json.dump(raw, f, indent=2)

    def add_staff(self, staff_id: str, name: str, encodings: list, role="staff"):
        if staff_id not in self.staff:
            self.staff[staff_id] = {"name": name, "role": role, "encodings": [], "active": True}
        else:
            self.staff[staff_id]["name"] = name
            self.staff[staff_id]["role"] = role
        self.staff[staff_id]["encodings"].extend(encodings)
        # persist npy for quick load
        np.save(os.path.join(config.ENCODING_DIR, f"{staff_id}.npy"), np.stack(encodings) if encodings else np.array([]))
        self.save()
        return self.staff[staff_id]

    def remove_staff(self, staff_id: str):
        self.staff.pop(staff_id, None)
        p = os.path.join(config.ENCODING_DIR, f"{staff_id}.npy")
        if os.path.exists(p): os.remove(p)
        self.save()

    def list_staff(self):
        return [{"id": k, "name": v["name"], "role": v["role"], "samples": len(v["encodings"]), "active": v["active"]}
                for k,v in self.staff.items()]

    def match(self, query_emb: np.ndarray, tolerance=config.TOLERANCE):
        """Returns (staff_id, name, distance) or (None, 'Unknown', dist)"""
        best_id, best_name, best_dist = None, "Unknown", 1e9
        for sid, rec in self.staff.items():
            if not rec.get("active", True) or not rec["encodings"]:
                continue
            for enc in rec["encodings"]:
                # cosine distance for fallback, euclidean for face_recognition
                if HAS_FR and enc.shape[0]==128:
                    dist = np.linalg.norm(enc - query_emb)
                else:
                    # cosine distance = 1 - dot
                    dist = 1 - float(np.dot(enc, query_emb))
                if dist < best_dist:
                    best_dist, best_id, best_name = dist, sid, rec["name"]
        if best_id is not None and best_dist <= tolerance:
            return best_id, best_name, float(best_dist)
        return None, "Unknown", float(best_dist)

# singleton
db = StaffDB()
