import cv2
import numpy as np

backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = True)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))



def motion_detection(frame, min_area=400):
    # 1. Background subtraction
    fg_mask = backSub.apply(frame)
    
    # 2. Keep only pure white (removes grey shadows)
    retval, mask_thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # 3. Opening: erode then dilate, removes small specks
    mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
    
    # 4. Find contours
    contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_contour_area = min_area
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    frame_out = frame.copy()
    for cnt in large_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        frame_out = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)
    
    # Display the resulting frame
    cv2.imshow('Frame_final', frame_out)

    



