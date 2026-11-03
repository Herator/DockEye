import sys

import cv2
import numpy as np

# Import function from other files File
from src.motionDetection import motion_detection, boxes_from_foreground, boat_backSub, pearson_backSub, pearson_kernel, boat_kernel
from src.zones import make_zones, draw_the_zone
from src.mediaPlayer import Player, collect

# -------------------------------------
#         GLOBAL VARIABLES
# -------------------------------------
MEDIA = "media/"
MD_ZONES_FILE = "config/MDZones.json"
MD_COLORS = {"person": (0, 0, 255), "boat": (255, 0, 0)}

# -------------------------------------
#               MEDIA
# -------------------------------------
paths = collect(sys.argv[1:] or [MEDIA])
if not paths:
    raise SystemExit(f"No videos found in {MEDIA!r}")

cam = cv2.VideoCapture(paths[0])
ret, first = cam.read()

# skip black frames at the start of the video
while ret and first.mean() < 10:
    ret, first = cam.read()
cam.release()
if not ret:
    raise SystemExit(f"{paths[0]} has no non-black frames")

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)


# ------------------------------------
#                ZONES
# ------------------------------------
md_zones = make_zones(first, MD_COLORS, MD_ZONES_FILE)
cv2.destroyWindow("Video")


# ------------------------------------
#          PER-FRAME DETECTION
# ------------------------------------
boat_settings = {"area": 40, "threshold": 50, "shadow_only": 1}
last_boat_mask = None
last_person_boxes = []


def reset_detection():
    global pearson_backSub, boat_backSub, last_boat_mask, last_person_boxes
    pearson_backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=20)
    boat_backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=50)
    last_boat_mask = None
    last_person_boxes = []


def detect(frame, reuse_mask=False):
    global last_boat_mask, last_person_boxes

    # Reuse the original foreground mask when changing sliders on a paused frame.
    if not reuse_mask or last_boat_mask is None:
        last_person_boxes = motion_detection(frame, md_zones["person"], 350, 150, pearson_backSub, pearson_kernel, blurre=False)
        blurred = cv2.GaussianBlur(frame, (21, 21), 0)
        last_boat_mask = boat_backSub.apply(blurred)

    boat_boxes, selected_mask = boxes_from_foreground(
        last_boat_mask, md_zones["boat"], boat_settings["area"],
        boat_settings["threshold"], boat_kernel, bool(boat_settings["shadow_only"])
    )
    cv2.imshow("Boat mask before threshold", last_boat_mask)
    cv2.imshow("Boat mask after threshold", selected_mask)

    for x, y, w, h in last_person_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)  # orange (BGR)

    for x, y, w, h in boat_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)  # green (BGR)

    # draw zones after detection so the lines aren't part of the motion input
    draw_the_zone(frame, md_zones, MD_COLORS)
    return frame


cv2.namedWindow("Boat tuning", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Boat tuning", 650, 260)
panel = np.zeros((100, 650, 3), dtype=np.uint8)
cv2.putText(panel, "Mode 0: threshold   |   Mode 1: grey only", (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(panel, "Threshold slider works in mode 0", (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
cv2.imshow("Boat tuning", panel)
cv2.createTrackbar("Minimum area", "Boat tuning", boat_settings["area"], 5000,
                   lambda value: boat_settings.__setitem__("area", value))
cv2.createTrackbar("Threshold (mode 0)", "Boat tuning", boat_settings["threshold"], 255,
                   lambda value: boat_settings.__setitem__("threshold", value))
cv2.createTrackbar("Mode 0 or 1", "Boat tuning", boat_settings["shadow_only"], 1,
                   lambda value: boat_settings.__setitem__("shadow_only", value))

player = Player(paths, detect,
                settings_key=lambda: tuple(boat_settings.values()),
                reprocess=lambda frame: detect(frame, reuse_mask=True),
                reset_detection=reset_detection)
player.toggles["boxes"] = True  # detection on by default, 'o' / Process button toggles it
player.run()
