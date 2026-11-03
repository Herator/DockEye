import cv2
import numpy as np



def motion_detection_mask(image, points):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    arr = np.array(points, np.int32)
    cv2.fillPoly(mask, [arr], 255)
            
    # draws the black bakgounds (nice to have for tubelshooting)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image
    
    

pearson_backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = True, varThreshold=20)
boat_backSub = cv2.createBackgroundSubtractorMOG2( detectShadows = True, varThreshold=50)


pearson_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
boat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))


def boxes_from_foreground(fg_mask, zone, min_area, threshold, kernel, shadow_only=False):
    """Apply the selected mask and return boxes and the mask before zoning."""
    if shadow_only:
        mask_thresh = cv2.inRange(fg_mask, 127, 127)
    else:
        _, mask_thresh = cv2.threshold(fg_mask, threshold, 255, cv2.THRESH_BINARY)
    masked_zone = motion_detection_mask(mask_thresh, zone)
    mask_merged = cv2.morphologyEx(masked_zone, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > min_area]
    return boxes, mask_thresh


def motion_detection(frame, zone, min_area, minValue, backSub, kernel, blurre=False):
    if blurre:
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
    fg_mask = backSub.apply(frame)
    boxes, _ = boxes_from_foreground(fg_mask, zone, min_area, minValue, kernel)
    return boxes


    



