"""Zone-based motion proposals; these are not semantic object labels."""
from dataclasses import dataclass

import cv2
import numpy as np


def motion_detection_mask(image, points):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, np.int32)], 255)
    return cv2.bitwise_and(image, image, mask=mask)


pearson_backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=20)
boat_backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=50)
pearson_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
boat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))


def motion_detection(frame, zone, min_area, minValue, backSub, kernel,
                     blurre=False, show_mask=False):
    if blurre:
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
    foreground = backSub.apply(frame)
    foreground[foreground == backSub.getShadowValue()] = 0
    _, mask = cv2.threshold(foreground, minValue, 255, cv2.THRESH_BINARY)
    mask = motion_detection_mask(mask, zone)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if show_mask:
        cv2.imshow('Mask merged', mask)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_area]


@dataclass
class MotionSettings:
    min_area: int
    threshold: int = 150
    variance: int = 20
    kernel_size: int = 3
    blur: bool = False


class MotionStage:
    def __init__(self, zones):
        self.zones = zones
        self.settings = {
            'person': MotionSettings(350),
            'boat': MotionSettings(1000, variance=50, kernel_size=10, blur=True),
        }
        self.backgrounds = {
            name: cv2.createBackgroundSubtractorMOG2(
                detectShadows=True, varThreshold=self.settings[name].variance)
            for name in zones
        }

    def process(self, frame):
        proposals = {}
        for name, zone in self.zones.items():
            settings = self.settings[name]
            background = self.backgrounds[name]
            background.setVarThreshold(settings.variance)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (settings.kernel_size, settings.kernel_size))
            proposals[name] = motion_detection(
                frame, zone, settings.min_area, settings.threshold,
                background, kernel, settings.blur)
        return proposals
