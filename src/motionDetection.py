import cv2
import numpy as np


# one per zone, the water needs a higher threshold (waves)
person_backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=20)
boat_backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=50)

person_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def motion_detection_mask(image, points):
    """Black out everything outside the zone polygon."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, np.int32)], 255)
    return cv2.bitwise_and(image, image, mask=mask)


def motion_detection(frame, zone, min_area, min_value, backSub, kernel, blur=False):
    """Boxes (x, y, w, h) of moving things in the zone bigger than min_area."""
    if blur:
        frame = cv2.GaussianBlur(frame, (21, 21), 0)

    fg_mask = backSub.apply(frame)

    # shadows are 127, min_value above that drops them
    _, mask_thresh = cv2.threshold(fg_mask, min_value, 255, cv2.THRESH_BINARY)
    masked_zone = motion_detection_mask(mask_thresh, zone)

    # join nearby blobs so one person is one box
    mask_merged = cv2.morphologyEx(masked_zone, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_area]
