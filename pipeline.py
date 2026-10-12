import cv2

from detector import in_zone, iou
from motionDetection import MotionStage
from tracker import ObjectTracker


def merge_regions(boxes, shape, padding=96):
    height, width = shape[:2]
    regions = []
    for x, y, w, h in boxes:
        left, top = max(0, x-padding), max(0, y-padding)
        right, bottom = min(width, x+w+padding), min(height, y+h+padding)
        if right > left and bottom > top:
            regions.append((left, top, right-left, bottom-top))
    # Merge overlapping crops so a model does not inspect the same area twice.
    changed = True
    while changed:
        changed = False
        for i in range(len(regions)):
            for j in range(i+1, len(regions)):
                if iou(regions[i], regions[j]) > 0:
                    a, b = regions[i], regions[j]
                    x, y = min(a[0], b[0]), min(a[1], b[1])
                    right, bottom = max(a[0]+a[2], b[0]+b[2]), max(a[1]+a[3], b[1]+b[3])
                    regions[i] = (x, y, right-x, bottom-y)
                    regions.pop(j)
                    changed = True
                    break
            if changed:
                break
    return regions


class Pipeline:
    def __init__(self, zones, detector=None, events=None, infer_interval=0.2,
                 warmup_seconds=1.0, max_gap=1.0):
        self.motion = MotionStage(zones)
        self.detector, self.events = detector, events
        self.tracker = ObjectTracker(max_gap=max_gap)
        self.infer_interval, self.warmup_seconds = infer_interval, warmup_seconds
        self.last_inference = float('-inf')
        self.start_time = None

    def process(self, frame, timestamp):
        if self.start_time is None:
            self.start_time = timestamp
        proposals = self.motion.process(frame)
        tracks = list(self.tracker.tracks.values())
        detections = []
        inference = False
        if (self.detector is not None and timestamp-self.start_time >= self.warmup_seconds
                and timestamp-self.last_inference >= self.infer_interval):
            boxes = [box for region in proposals.values() for box in region]
            # Keep classifying active tracks even when they stop moving.
            boxes += [track.box for track in tracks if timestamp-track.last_seen <= self.tracker.max_gap]
            regions = merge_regions(boxes, frame.shape)
            if regions:
                detections = [d for d in self.detector.detect(frame, regions)
                              if in_zone(d, self.motion.zones)]
                inference = True
                self.last_inference = timestamp
        tracks = self.tracker.update(detections, timestamp, observed=inference)
        if self.events is not None:
            self.events.update(tracks, timestamp, frame, inference=inference)
        return proposals, tracks, detections
