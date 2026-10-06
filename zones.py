import cv2
import json
import os
import numpy as np


def make_zones(image, zone_colors, zones_file):
    if os.path.exists(zones_file):
        zones = json.load(open(zones_file))
        
    else:
        zones = {}
        for name in zone_colors:
            points = []
            cv2.setMouseCallback("Video", lambda e, x, y, f, p: points.append((x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
            while True:
                img = image.copy()
                for p in points:
                    cv2.circle(img, p, 6, zone_colors[name], -1)
                cv2.putText(img, f"Click {name} corners, Enter when done", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.imshow("Video", img)
                if cv2.waitKey(20) == 13 and len(points) >= 3:
                    break
            zones[name] = points
        json.dump(zones, open(zones_file, "w"))
    return zones

def draw_the_zone(frame, zones, colors):
    for name, pts in zones.items():
            cv2.polylines(frame, [np.array(pts, np.int32)], True, colors[name], 4)