import cv2
import numpy as np

# Import function from other files File
from motionDetection import motion_detection, motion_detection_mask, boat_backSub, pearson_backSub, pearson_kernel, boat_kernel
from zones import make_zones, draw_the_zone

# -------------------------------------
#         GLOBAL VARIABLES
# -------------------------------------
VIDEO = "media/Oscar.mp4"
MD_ZONES_FILE = "config/MDZones.json"
MD_COLORS = {"person": (0, 0, 255), "boat": (255, 0, 0)}

# -------------------------------------
#               MEDIA
# -------------------------------------
cam = cv2.VideoCapture(VIDEO)
fps = cam.get(cv2.CAP_PROP_FPS) or 30
ret, first = cam.read()

# skip black frames at the start of the video
while ret and first.mean() < 10:
    ret, first = cam.read()
if not ret:
    raise SystemExit(f"{VIDEO} has no non-black frames")

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)


# ------------------------------------
#                ZONES
# ------------------------------------    
md_zones = make_zones(first, MD_COLORS, MD_ZONES_FILE)




# Play with zones drawn on top
while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    #motion detection function:  WITHOUT different zones
    #motion_detection(frame) 
    
    #person_view = motion_detection_mask(first, md_zones["person"])
    #cv2.imshow("Person zone", person_view) #SHOW MASK for where to not search for person
    
    #boat_view = motion_detection_mask(first, md_zones["boat"])
    #cv2.imshow("boat zone", boat_view) #SHOW MASK for where to not search for boat
    person_boxes = motion_detection(frame, md_zones["person"], 350, 150, pearson_backSub, pearson_kernel, blurre=False)

    boat_boxes = motion_detection(frame, md_zones["boat"], 1000, 50, boat_backSub, boat_kernel, blurre=True)
    

    for x, y, w, h in person_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)  # orange (BGR)

    for x, y, w, h in boat_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)  # green (BGR)

    # draw zones after detection so the lines aren't part of the motion input
    draw_the_zone(frame, md_zones, MD_COLORS)

    cv2.imshow("Video", frame)
    
    if cv2.waitKey(int(1000 / fps)) == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()