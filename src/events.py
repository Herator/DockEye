import csv
import os

import cv2

FIELDS = ["video", "class", "id", "event", "start_s", "end_s", "duration_s", "snapshot"]
EVENT_NAME = {"person": "person_on_dock", "boat": "boat_passed"}


class EventLog:
    """Turns tracks into events in a CSV file (FR16-17).

    Seen for min_seconds: event, one snapshot. Gone for gap_seconds: CSV row written,
    so a short miss doesn't split one event in two.
    """

    def __init__(self, csv_path="output/events.csv", snapshot_dir="output/snapshots", min_seconds=2.0, gap_seconds=1.0):
        self.csv_path, self.snapshot_dir = csv_path, snapshot_dir
        self.min_seconds, self.gap_seconds = min_seconds, gap_seconds
        self.video = ""
        self.active = {}   # (cls, id) -> start, last, snapshot
        self.count = {}    # cls -> events in this video (FR18)

    def start_video(self, video):
        self.close()
        self.video, self.count = video, {}

    def update(self, cls, tracks, frame, t):
        for tid, (x, y, w, h) in tracks.items():
            ev = self.active.setdefault((cls, tid), {"start": t, "last": t, "snapshot": ""})
            ev["last"] = t
            if not ev["snapshot"] and t - ev["start"] >= self.min_seconds:
                ev["snapshot"] = self._snapshot(cls, tid, frame, (x, y, w, h), t)
                self.count[cls] = self.count.get(cls, 0) + 1
        for key in [k for k, ev in self.active.items()
                    if k[0] == cls and k[1] not in tracks and t - ev["last"] > self.gap_seconds]:
            self._end(key)

    def close(self):
        """End all open events."""
        for key in list(self.active):
            self._end(key)

    def _end(self, key):
        ev = self.active.pop(key)
        if not ev["snapshot"]:
            return  # too short to count as an event
        cls, tid = key
        new = not os.path.exists(self.csv_path)
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(FIELDS)
            w.writerow([os.path.basename(self.video), cls, tid, EVENT_NAME.get(cls, cls),
                        f"{ev['start']:.1f}", f"{ev['last']:.1f}", f"{ev['last'] - ev['start']:.1f}",
                        ev["snapshot"]])

    def _snapshot(self, cls, tid, frame, box, t):
        os.makedirs(self.snapshot_dir, exist_ok=True)
        name = f"{os.path.splitext(os.path.basename(self.video))[0]}_{cls}{tid}_{t:.1f}s.jpg"
        path = os.path.join(self.snapshot_dir, name)
        img = frame.copy()
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite(path, img)
        return path
