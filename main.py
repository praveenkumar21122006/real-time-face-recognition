import cv2
import numpy as np
import os, time, json, base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import config
import recognition
from recognition import db
from entry_logger import logger
from camera import manager

app = FastAPI(title="Real-Time Face Recognition Entry System")

# --- Lifecycle ---
@app.on_event("startup")
def startup():
    # don't auto-start cameras if no webcam; lazy start on /video_feed
    pass

@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html","r") as f:
        return f.read()

# --- Staff CRUD ---
@app.get("/api/staff")
def list_staff():
    return db.list_staff()

@app.post("/api/staff/enroll")
async def enroll_staff(
    staff_id: str = Form(...),
    name: str = Form(...),
    role: str = Form("staff"),
    files: list[UploadFile] = File(...)
):
    """
    Enroll staff with 1-5 face images. Each image -> one embedding.
    Instant matching after enroll.
    """
    encodings = []
    for f in files:
        data = await f.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None: continue
        boxes = recognition.detect_faces(img)
        if not boxes:
            # try whole image as face
            h,w = img.shape[:2]
            boxes = [(0,0,w,h)]
        for box in boxes[:1]:  # one face per image
            emb = recognition.encode_face(img, box)
            if emb is not None:
                encodings.append(emb)
        # save thumb
        thumb_path = os.path.join(config.THUMB_DIR, f"{staff_id}.jpg")
        cv2.imwrite(thumb_path, cv2.resize(img, (160,160)))

    if not encodings:
        raise HTTPException(400, "No face detected in uploaded images. Try clearer frontal photos.")

    rec = db.add_staff(staff_id, name, encodings, role)
    return {"ok": True, "staff_id": staff_id, "name": name, "samples": len(encodings)}

@app.delete("/api/staff/{staff_id}")
def delete_staff(staff_id: str):
    db.remove_staff(staff_id)
    return {"ok": True}

@app.post("/api/staff/{staff_id}/toggle")
def toggle_staff(staff_id: str):
    rec = db.staff.get(staff_id)
    if not rec: raise HTTPException(404, "not found")
    rec["active"] = not rec.get("active", True)
    db.save()
    return {"active": rec["active"]}

# --- Recognition ---
@app.post("/api/recognize")
async def recognize_single(file: UploadFile = File(...)):
    """Instant single-image check (useful for testing / door intercom snapshot)."""
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400, "invalid image")
    boxes = recognition.detect_faces(img)
    if not boxes:
        h,w = img.shape[:2]
        boxes = [(0,0,w,h)]
    results = []
    for box in boxes:
        emb = recognition.encode_face(img, box)
        if emb is None: continue
        sid, name, dist = db.match(emb)
        x,y,w,h = box
        results.append({"box":[int(x),int(y),int(w),int(h)], "staff_id": sid, "name": name, "distance": float(dist), "granted": sid is not None})
        # log
        if sid:
            logger.log_entry(sid, name, "api_test", "GRANTED", dist)
        else:
            logger.log_entry("unknown", "Unknown", "api_test", "DENIED", dist)
    return {"faces": results, "count": len(results)}

@app.get("/api/log")
def get_log(n: int = 50):
    return logger.recent(n)

# --- Video streaming ---
def gen_frames(entrance: str):
    cam = manager.get(entrance)
    if not cam:
        # lazy start single cam
        src = config.CAMERA_SOURCES.get(entrance, 0)
        from camera import CameraStream
        cam = CameraStream(src, entrance).start()
        manager.cams[entrance] = cam
        time.sleep(0.8)
    while True:
        frame = cam.read(timeout=2)
        if frame is None:
            # placeholder
            frame = np.zeros((480,640,3), dtype=np.uint8)
            cv2.putText(frame, "No camera signal - check CAMERA_SOURCES", (20,240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok: continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.03)

@app.get("/video_feed/{entrance}")
def video_feed(entrance: str):
    return StreamingResponse(gen_frames(entrance), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/entrances")
def entrances():
    return list(config.CAMERA_SOURCES.keys())

@app.get("/health")
def health():
    return {"status":"ok", "staff": len(db.staff), "has_face_recognition": recognition.HAS_FR, "entrances": list(config.CAMERA_SOURCES.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
