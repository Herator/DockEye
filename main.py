import sys

import cv2

# Import function from other files File
from src.motionDetection import motion_detection, motion_detection_mask, boat_backSub, pearson_backSub, pearson_kernel, boat_kernel
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
def detect(frame):
    #person_view = motion_detection_mask(first, md_zones["person"])
    #cv2.imshow("Person zone", person_view) #SHOW MASK for where to not search for person

    person_boxes = motion_detection(frame, md_zones["person"], 350, 150, pearson_backSub, pearson_kernel, blurre=False)
    boat_boxes = motion_detection(frame, md_zones["boat"], 1000, 50, boat_backSub, boat_kernel, blurre=True)

    for x, y, w, h in person_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)  # orange (BGR)

    for x, y, w, h in boat_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)  # green (BGR)

    # draw zones after detection so the lines aren't part of the motion input
    draw_the_zone(frame, md_zones, MD_COLORS)
    return frame


player = Player(paths, detect)
player.toggles["boxes"] = True  # detection on by default, 'o' / Process button toggles it
player.run()
