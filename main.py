import cv2
import numpy as np

# Import function from other files File
from motionDetection import motion_detection, motion_detection_mask, boat_backSub, pearson_backSub, pearson_kernel, boat_kernel
from zones import make_zones, draw_the_zone

# -------------------------------------
#         GLOBAL VARIABLES
# -------------------------------------
VIDEO = "media/nameOfVideo.mp4"
ZONES_FILE = "config/zones.json"
COLORS = {"dock": (0, 165, 255), "water": (255, 100, 0)}  # BGR

MD_ZONES_FILE = "config/MDZones.json"
MD_COLORS = {"person": (0, 0, 255), "boat": (255, 0, 0)}

# -------------------------------------
#               MEDIA
# -------------------------------------
cam = cv2.VideoCapture(VIDEO)
fps = cam.get(cv2.CAP_PROP_FPS) or 30
ret, first = cam.read()

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)


# ------------------------------------
#                ZONES
# ------------------------------------    
basic_zones = make_zones(first, COLORS, ZONES_FILE)
md_zones = make_zones(first, MD_COLORS, MD_ZONES_FILE)


cam.set(cv2.CAP_PROP_POS_FRAMES, 0)


# Play with zones drawn on top
while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    #motion detection function:  WITHOUT different zones
    #motion_detection(frame) 
    
    # draw Dock and water zones
    #draw_the_zone(frame, basic_zones, COLORS)
       
    # draw motion zones  
    #draw_the_zone(frame, md_zones, MD_COLORS)  
    
    #person_view = motion_detection_mask(first, md_zones["person"])
    #cv2.imshow("Person zone", person_view) #SHOW MASK for where to not search for person
    
    #boat_view = motion_detection_mask(first, md_zones["boat"])
    #cv2.imshow("boat zone", boat_view) #SHOW MASK for where to not search for boat
    person_boxes = motion_detection(frame, md_zones["person"], 350, 150, pearson_backSub, pearson_kernel, blurre=False)

    boat_boxes = motion_detection(frame, md_zones["boat"], 1000, 50, boat_backSub, boat_kernel, blurre=True)
    

    for x, y, w, h in person_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 200), 2)
        
    for x, y, w, h in boat_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 200), 2)
    
    cv2.imshow("Video", frame)
    
    if cv2.waitKey(int(1000 / fps)) == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()