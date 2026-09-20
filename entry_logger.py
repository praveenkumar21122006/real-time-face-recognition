import json, os, time
from datetime import datetime
import config

class EntryLogger:
    def __init__(self, path=config.LOG_PATH):
        self.path = path
        self.entries = []
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path,"r") as f:
                    self.entries = json.load(f)
            except: self.entries = []

    def save(self):
        with open(self.path,"w") as f:
            json.dump(self.entries, f, indent=2)

    def log_entry(self, staff_id, name, entrance, status, confidence=None):
        rec = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "staff_id": staff_id,
            "name": name,
            "entrance": entrance,
            "status": status,  # GRANTED / DENIED / UNKNOWN
            "confidence": confidence
        }
        self.entries.insert(0, rec)
        self.entries = self.entries[:2000]
        self.save()
        print(f"[ENTRY] {rec['timestamp']} {entrance} {status} {name} ({staff_id})")
        return rec

    def recent(self, n=50):
        return self.entries[:n]

logger = EntryLogger()
