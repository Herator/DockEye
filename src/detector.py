import os

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from ultralytics.trackers import track as yolo_track
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.utils import matching

PERSON, BOAT = 0, 8         # COCO class ids
MODEL = "model/yolo26s.pt"  # 'n' missed people far out on the dock
BOAT_MOVE_PX = 40           # ignores moored boats
TRACKER = "config/tracker.yaml"
ORT_THREADS = max(2, (os.cpu_count() or 4) // 4)  # leave the rest for the video


class DockTracker(BYTETracker):
    """ByteTrack, but new tracks are confirmed with match_thresh instead of a fixed 0.3 overlap.

    Distant boats move almost their own width between YOLO runs, so they never got an ID.
    NOTE: overrides a private method in Ultralytics 8.4, check it if you upgrade.
    """

    def _unconfirmed_association(self, unconfirmed, u_detection, detections, activated, removed):
        detections = [detections[i] for i in u_detection]
        if not unconfirmed:
            return list(range(len(detections))), detections
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            unconfirmed[it].mark_removed()
            removed.append(unconfirmed[it])
        return u_detection, detections


yolo_track.TRACKER_MAP["dockeye"] = DockTracker  # tracker_type: dockeye in tracker.yaml


def onnx_model(pt_path, h, w):
    """YOLO as ONNX, about 2x faster than PyTorch. Exported once per crop size (multiple of 32)."""
    imgsz = [-(-h // 32) * 32, -(-w // 32) * 32]
    path = f"{os.path.splitext(pt_path)[0]}_{imgsz[0]}x{imgsz[1]}.onnx"
    if not os.path.exists(path):
        os.replace(YOLO(pt_path).export(format="onnx", imgsz=imgsz, simplify=True), path)
    return YOLO(path, task="detect"), imgsz


def on_dock(box, dock_zone):
    """True if the feet (bottom middle of the box) are on the dock."""
    x, y, w, h = box
    return cv2.pointPolygonTest(np.array(dock_zone, np.int32), (x + w / 2, y + h), False) >= 0


class Detector:
    """YOLO plus tracking. One per video so the IDs start over."""

    def __init__(self, zones, model=MODEL):
        self.dock_zone = zones["person"]
        # YOLO only gets the rectangle around the zones
        pts = np.array([p for zone in zones.values() for p in zone])
        self.x0, self.y0 = map(int, pts.min(0).clip(0))
        self.x1, self.y1 = map(int, pts.max(0) + 1)
        self.model, self.imgsz = onnx_model(model, self.y1 - self.y0, self.x1 - self.x0)
        self.first_centre = {}  # boat id -> first centre

    def track(self, frame):
        """{"person": {id: box}, "boat": {id: box}}. Only people on the dock and boats that moved."""
        r = self.model.track(frame[self.y0:self.y1, self.x0:self.x1], persist=True, classes=[PERSON, BOAT],
                             imgsz=self.imgsz, tracker=TRACKER, verbose=False)[0]
        self._limit_threads()

        out = {"person": {}, "boat": {}}
        if r.boxes.id is None:
            return out
        for (x1, y1, x2, y2), cls, tid in zip(r.boxes.xyxy.int().tolist(), r.boxes.cls.int().tolist(),
                                              r.boxes.id.int().tolist()):
            box = (x1 + self.x0, y1 + self.y0, x2 - x1, y2 - y1)  # back to full frame
            if cls == PERSON and on_dock(box, self.dock_zone):
                out["person"][tid] = box
            elif cls == BOAT and self._moved(tid, box):
                out["boat"][tid] = box
        return out

    def _limit_threads(self):
        # by default ONNX Runtime busy-waits on every core and the video stutters.
        # NOTE: swaps Ultralytics' private session (8.4, CPU only), check it if you upgrade.
        backend = self.model.predictor.model.backend
        if getattr(backend, "limited", False):
            return
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = ORT_THREADS
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        backend.session = ort.InferenceSession(self.model.ckpt_path, opts, providers=["CPUExecutionProvider"])
        backend.limited = True

    def _moved(self, tid, box):
        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2
        fx, fy = self.first_centre.setdefault(tid, (cx, cy))
        return np.hypot(cx - fx, cy - fy) >= BOAT_MOVE_PX
