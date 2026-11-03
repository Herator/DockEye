import os

from ultralytics import YOLO

MODEL = "model/fishing_cls.pt"  # made by dataset/train_fishing.py


def crop(frame, box, pad=0.75):
    """Person box grown on every side so the rod is in the crop."""
    x, y, w, h = box
    p = int(max(w, h) * pad)
    H, W = frame.shape[:2]
    return frame[max(0, y - p):min(H, y + h + p), max(0, x - p):min(W, x + w + p)]


class FishingClassifier:
    """Says fishing / not_fishing for a person crop. Does nothing until the model is trained."""

    def __init__(self, model=MODEL):
        self.model = YOLO(model) if os.path.exists(model) else None

    def is_fishing(self, frame, box):
        if self.model is None:
            return False
        r = self.model.predict(crop(frame, box), verbose=False)[0]
        return r.names[r.probs.top1] == "fishing"
