import cv2
import numpy as np


class Tracker:
    """Keeps a box on an object after it stops moving.

    Motion boxes start or move tracks. A track that gets no motion stays as long
    as its area still looks different from the empty scene (self.ref).
    The empty scene is learned from the MOG2 background, except inside track
    boxes, so a person standing still never becomes part of it.
    """

    def __init__(self, diff_threshold=30, min_changed_px=200, pad=20, min_aspect=0):
        self.tracks = {}                      # id -> (x, y, w, h)
        self.next_id = 1
        self.ref = None                       # blurred grey image of the empty scene
        self.diff_threshold = diff_threshold  # grey-level change that counts as "different"
        self.min_changed_px = min_changed_px  # changed pixels needed to count as "something is there"
        self.pad = pad                        # how close (px) a motion box must be to belong to a track
        self.min_aspect = min_aspect          # new tracks only from boxes with height/width >= this

    def update(self, frame, boxes, background):
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        bg = cv2.GaussianBlur(cv2.cvtColor(background, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        if self.ref is None:
            self.ref = bg

        # ignore "ghosts": motion where the image looks like the empty scene
        boxes = [b for b in boxes if self._something_there(gray, b)]

        moved = {}
        for box in boxes:
            tid = self._match(box)
            if tid is None:
                if box[3] < self.min_aspect * box[2]:
                    continue   # wrong shape to be a new object (e.g. shadow, moored boat)
                tid = self.next_id
                self.next_id += 1
            moved[tid] = _union(moved[tid], box) if tid in moved else box

        for tid, box in list(self.tracks.items()):
            if tid not in moved and not self._something_there(gray, box):
                del self.tracks[tid]   # no motion and looks empty -> they left
        self.tracks.update(moved)

        # learn the empty scene everywhere except where a track is
        tracked = np.zeros(gray.shape, bool)
        for x, y, w, h in self.tracks.values():
            tracked[y:y + h, x:x + w] = True
        self.ref = np.where(tracked, self.ref, bg)

        return self.tracks

    def _match(self, box):
        best, best_overlap = None, 0
        for tid, t in self.tracks.items():
            overlap = _overlap(_grow(t, self.pad), box)
            if overlap > best_overlap:
                best, best_overlap = tid, overlap
        return best

    def _something_there(self, gray, box):
        x, y, w, h = box
        diff = cv2.absdiff(gray[y:y + h, x:x + w], self.ref[y:y + h, x:x + w])
        return np.count_nonzero(diff > self.diff_threshold) > self.min_changed_px


def _grow(b, p):
    x, y, w, h = b
    return (x - p, y - p, w + 2 * p, h + 2 * p)


def _overlap(a, b):
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)


def _union(a, b):
    x, y = min(a[0], b[0]), min(a[1], b[1])
    return (x, y, max(a[0] + a[2], b[0] + b[2]) - x, max(a[1] + a[3], b[1] + b[3]) - y)


# Detector-driven tracker used by the application. The earlier empty-scene
# Tracker above remains available for experiments but is not a Kalman filter.
from dataclasses import dataclass
from detector import family, iou


@dataclass
class ObjectTrack:
    id: int
    box: tuple
    label: str
    confidence: float
    first_seen: float
    last_seen: float
    observed: bool = True


class ObjectTracker:
    """One-to-one association with a bounded grace period for missed detections.

    Person/fishing class changes keep the same ID. This does not guarantee
    re-identification after occlusion or leaving the frame.
    """
    def __init__(self, max_gap=1.0, match_iou=0.1, max_distance=80):
        self.max_gap, self.match_iou, self.max_distance = max_gap, match_iou, max_distance
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, timestamp, observed=True):
        self.tracks = {tid: track for tid, track in self.tracks.items()
                       if timestamp-track.last_seen <= self.max_gap}
        for track in self.tracks.values():
            track.observed = False
        if not observed:
            return list(self.tracks.values())
        pairs = []
        for index, detection in enumerate(detections):
            x, y, w, h = detection.box
            for tid, track in self.tracks.items():
                if family(track.label) != family(detection.label):
                    continue
                tx, ty, tw, th = track.box
                overlap = iou(track.box, detection.box)
                distance = ((x+w/2-tx-tw/2)**2 + (y+h/2-ty-th/2)**2)**0.5
                if overlap >= self.match_iou or distance <= self.max_distance:
                    pairs.append((overlap-distance/10000, index, tid))
        used_detections, used_tracks = set(), set()
        for _, index, tid in sorted(pairs, reverse=True):
            if index in used_detections or tid in used_tracks:
                continue
            detection, track = detections[index], self.tracks[tid]
            track.box, track.label, track.confidence = detection.box, detection.label, detection.confidence
            track.last_seen, track.observed = timestamp, True
            used_detections.add(index)
            used_tracks.add(tid)
        for index, detection in enumerate(detections):
            if index not in used_detections:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = ObjectTrack(
                    tid, detection.box, detection.label, detection.confidence,
                    timestamp, timestamp)
        return list(self.tracks.values())

