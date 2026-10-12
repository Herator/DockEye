"""Experimental geometric fishing candidates, not proof of fishing activity."""
import math

import cv2
import numpy as np

from detector import Detection


class RodHeuristic:
    def __init__(self, minimum_length=0.4, angle_min=10, angle_max=80):
        self.minimum_length = minimum_length
        self.angle_min, self.angle_max = angle_min, angle_max

    def candidate_line(self, frame, box):
        x, y, w, h = box
        height, width = frame.shape[:2]
        padding = int(h)
        left, top = max(0,x-padding), max(0,y-padding//2)
        right, bottom = min(width,x+w+padding), min(height,y+h+padding//2)
        crop = frame[top:bottom,left:right]
        if crop.size == 0 or h <= 0 or w <= 0:
            return None
        gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,60,160)
        lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=20,
                                minLineLength=max(15,int(h*self.minimum_length)), maxLineGap=8)
        if lines is None:
            return None
        candidates = []
        for raw in lines[:,0]:
            ax, ay, bx, by = map(float,raw)
            ax, bx, ay, by = ax+left, bx+left, ay+top, by+top
            length = math.hypot(bx-ax,by-ay)
            angle = math.degrees(math.atan2(abs(by-ay),abs(bx-ax)))
            if length < h*self.minimum_length or not self.angle_min <= angle <= self.angle_max:
                continue
            def near_hands(px,py):
                return x-0.2*w <= px <= x+1.2*w and y+0.2*h <= py <= y+0.85*h
            def extends_away(px,py):
                return px < x-0.2*h or px > x+w+0.2*h
            if ((near_hands(ax,ay) and extends_away(bx,by)) or
                    (near_hands(bx,by) and extends_away(ax,ay))):
                candidates.append((length,(int(ax),int(ay),int(bx),int(by))))
        return max(candidates,default=(0,None))[1]


class RodAssistedDetector:
    """Preserves semantic person detection; adds only a tentative rod signal."""
    has_fishing = False
    has_fishing_candidates = True

    def __init__(self, detector, heuristic=None):
        self.detector = detector
        self.heuristic = heuristic or RodHeuristic()

    @property
    def calls(self):
        return self.detector.calls

    def detect(self, frame, regions):
        detections = []
        for detection in self.detector.detect(frame,regions):
            if detection.label in ('person','person_fishing'):
                label = ('person_fishing_candidate' if self.heuristic.candidate_line(frame,detection.box)
                         is not None else 'person')
                detection = Detection(detection.box,label,detection.confidence)
            detections.append(detection)
        return detections
