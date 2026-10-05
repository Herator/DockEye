import cv2
import json
import os
import numpy as np

VIDEO = "media/IMG_6660.MOV"
ZONES_FILE = "zones.json"
COLORS = {"dock": (0, 165, 255), "water": (255, 100, 0)}  # BGR

cam = cv2.VideoCapture(VIDEO)
fps = cam.get(cv2.CAP_PROP_FPS) or 30
ret, first = cam.read()

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)


# Draw zones once (click corners, Enter = next zone)
if os.path.exists(ZONES_FILE):
    zones = json.load(open(ZONES_FILE))
else:
    zones = {}
    for name in COLORS:
        points = []
        cv2.setMouseCallback("Video", lambda e, x, y, f, p: points.append((x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
        while True:
            img = first.copy()
            for p in points:
                cv2.circle(img, p, 6, COLORS[name], -1)
            cv2.putText(img, f"Click {name} corners, Enter when done", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.imshow("Video", img)
            if cv2.waitKey(20) == 13 and len(points) >= 3:
                break
        zones[name] = points
    json.dump(zones, open(ZONES_FILE, "w"))

cam.set(cv2.CAP_PROP_POS_FRAMES, 0)



# Play with zones drawn on top
while True:
    ret, frame = cam.read()
    if not ret:
        break
    for name, pts in zones.items():
        cv2.polylines(frame, [np.array(pts, np.int32)], True, COLORS[name], 4)
    cv2.imshow("Video", frame)
    if cv2.waitKey(int(1000 / fps)) == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()