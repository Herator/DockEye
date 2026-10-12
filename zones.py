import json
from pathlib import Path

import cv2
import numpy as np


def validate_zones(zones, image=None):
    if not isinstance(zones, dict) or set(zones) != {'person', 'boat'}:
        raise ValueError('Zones must contain exactly person and boat polygons')
    for name, points in zones.items():
        arr = np.asarray(points, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
            raise ValueError(f'{name}: need at least three (x, y) corners')
        if not np.isfinite(arr).all() or (arr < 0).any():
            raise ValueError(f'{name}: invalid coordinates')
        if image is not None:
            height, width = image.shape[:2]
            if (arr[:, 0] >= width).any() or (arr[:, 1] >= height).any():
                raise ValueError(f'{name}: zones do not fit {width}x{height}; use another zones file')
        if cv2.contourArea(arr.astype(np.float32)) == 0:
            raise ValueError(f'{name}: polygon has no area')
    return zones


def make_zones(image, zone_colors, zones_file, headless=False, reference_size=None):
    path = Path(zones_file)
    if path.exists():
        zones = json.loads(path.read_text())
        if reference_size is not None:
            ref_width, ref_height = reference_size
            if min(ref_width,ref_height) <= 0:
                raise ValueError('Zone reference dimensions must be positive')
            validate_zones(zones)
            height,width = image.shape[:2]
            zones = {name: [[min(width-1,round(x*width/ref_width)),
                              min(height-1,round(y*height/ref_height))] for x,y in points]
                     for name,points in zones.items()}
        return validate_zones(zones, image)
    if headless:
        raise ValueError(f'Cannot click zones in headless mode; create {path} first')
    zones = {}
    for name in zone_colors:
        points = []
        def click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
        cv2.setMouseCallback('Video', click)
        while True:
            img = image.copy()
            for point in points:
                cv2.circle(img, point, 6, zone_colors[name], -1)
            cv2.putText(img, f'Click {name} corners; Enter saves; q cancels', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Video', img)
            key = cv2.waitKey(20) & 255
            if key == ord('q'):
                raise ValueError('Zone setup cancelled')
            if key == 13 and len(points) >= 3:
                break
        zones[name] = points
    validate_zones(zones, image)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(zones))
    return zones


def draw_the_zone(frame, zones, colors):
    for name, points in zones.items():
        cv2.polylines(frame, [np.array(points, np.int32)], True, colors[name], 2)
