"""Ultralytics adapter. Custom fishing weights must be trained separately."""
from dataclasses import dataclass
from pathlib import Path

import cv2


CLASSES = ('boat', 'person', 'person_fishing')


@dataclass(frozen=True)
class Detection:
    box: tuple
    label: str
    confidence: float


def family(label):
    return 'person' if label in ('person_fishing', 'person_fishing_candidate') else label


def iou(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    intersection = max(0, right-x) * max(0, bottom-y)
    union = a[2]*a[3] + b[2]*b[3] - intersection
    return intersection / union if union else 0


def suppress_duplicates(detections, threshold=0.5):
    kept = []
    for detection in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if not any(family(d.label) == family(detection.label) and
                   iou(d.box, detection.box) > threshold for d in kept):
            kept.append(detection)
    return kept


class ModelDetector:
    def __init__(self, weights, confidence=0.5, device='cpu'):
        if not Path(weights).is_file():
            raise ValueError(f'Model weights not found: {weights}. Train or supply a model first.')
        from ultralytics import YOLO
        import torch
        torch.set_num_threads(2)
        self.model = YOLO(str(weights))
        self.confidence, self.device = confidence, device
        self.names = {key: value.lower().replace(' ', '_')
                      for key, value in self.model.names.items()}
        if not {'person', 'boat'}.issubset(set(self.names.values())):
            raise ValueError('Model must include person and boat classes')
        self.has_fishing = 'person_fishing' in self.names.values()
        self.calls = 0

    def detect(self, frame, regions):
        results = []
        height, width = frame.shape[:2]
        for x, y, w, h in regions:
            x, y = max(0, int(x)), max(0, int(y))
            w, h = min(int(w), width-x), min(int(h), height-y)
            if w <= 0 or h <= 0:
                continue
            self.calls += 1
            prediction = self.model.predict(
                source=frame[y:y+h, x:x+w], conf=self.confidence,
                device=self.device, verbose=False)[0]
            for box in prediction.boxes:
                label = self.names[int(box.cls.item())]
                if label not in CLASSES:
                    continue
                left, top, right, bottom = box.xyxy[0].cpu().tolist()
                results.append(Detection(
                    (int(left)+x, int(top)+y, max(1, int(right-left)), max(1, int(bottom-top))),
                    label, float(box.conf.item())))
        return suppress_duplicates(results)


def in_zone(detection, zones):
    import numpy as np
    zone_name = family(detection.label)
    if zone_name not in zones:
        return False
    x, y, w, h = detection.box
    point = (float(x+w/2), float(y+h if zone_name == 'person' else y+h/2))
    return cv2.pointPolygonTest(np.array(zones[zone_name], np.float32), point, False) >= 0
