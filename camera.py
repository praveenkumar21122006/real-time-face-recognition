import cv2
import time
import threading
import queue
import numpy as np
from collections import deque
import config
import recognition

class CameraStream:
    def __init__(self, source, name="entrance"):
        self.source = source
        self.name = name
        self.cap = None
        self.running = False
        self.q = queue.Queue(maxsize=2)
        self.thread = None
        self.last_results = []  # [(box, id, name, dist)]
        self.fps = 0
        self._frame_count = 0
        self._cooldown = {}  # id -> last trigger time

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        # RTSP / cam tuning
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.running = False
        if self.thread: self.thread.join(timeout=2)
        if self.cap: self.cap.release()

    def _loop(self):
        t0 = time.time()
        frames = 0
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.1)
                # try reconnect
                self.cap.release()
                time.sleep(0.5)
                self.cap = cv2.VideoCapture(self.source)
                continue

            self._frame_count += 1
            # process every Nth frame
            if self._frame_count % config.FRAME_SKIP == 0:
                results = self._recognize(frame)
                self.last_results = results
                # throttle repeat triggers
                now = time.time()
                for _, sid, _, _ in results:
                    if sid and sid in self._cooldown and now - self._cooldown[sid] < config.RECOGNITION_COOLDOWN:
                        continue
                    if sid:
                        self._cooldown[sid] = now
                        self._on_match(sid, results)

            # put latest frame for streaming (drop old)
            if not self.q.empty():
                try: self.q.get_nowait()
                except: pass
            # overlay
            vis = self._draw(frame, self.last_results)
            self.q.put(vis)

            frames += 1
            if time.time() - t0 >= 1.0:
                self.fps = frames
                frames = 0
                t0 = time.time()

    def _recognize(self, frame):
        boxes = recognition.detect_faces(frame)
        out = []
        for box in boxes:
            emb = recognition.encode_face(frame, box)
            if emb is None: continue
            sid, name, dist = recognition.db.match(emb)
            out.append((box, sid, name, dist))
        return out

    def _on_match(self, staff_id, results):
        # hook: log + door unlock
        from entry_logger import logger
        rec = recognition.db.staff.get(staff_id)
        if rec:
            logger.log_entry(staff_id, rec["name"], self.name, "GRANTED")

    def _draw(self, frame, results):
        for (x,y,w,h), sid, name, dist in results:
            color = (0,255,0) if sid else (0,0,255)
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            label = f"{name} {dist:.2f}" if sid else f"Unknown {dist:.2f}"
            cv2.rectangle(frame, (x,y-22), (x+180,y), color, -1)
            cv2.putText(frame, label, (x+5,y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, f"{self.name} | FPS:{self.fps} | {recognition.HAS_FR and 'face_recognition' or 'OpenCV fallback'}",
                    (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        return frame

    def read(self, timeout=1):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

# Manager for multiple entrances
class CameraManager:
    def __init__(self):
        self.cams = {}

    def start_all(self):
        for name, src in config.CAMERA_SOURCES.items():
            c = CameraStream(src, name).start()
            self.cams[name] = c

    def get(self, name):
        return self.cams.get(name)

    def stop_all(self):
        for c in self.cams.values(): c.stop()

manager = CameraManager()
