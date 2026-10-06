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


def motion_detection(frame, zone, min_area, minValue, backSub, kernel, blurre=False):
    
    if blurre:
        frame = cv2.GaussianBlur(frame,(21, 21),0)
     
    # 1. Background subtraction
    fg_mask = backSub.apply(frame)
    
    # 2. Keep only pure white (removes grey shadows)
    _, mask_thresh = cv2.threshold(fg_mask, minValue, 255, cv2.THRESH_BINARY) # cv2.threshold(fg_mask, replace all values under x to 0, 255, cv2.THRESH_BINARY )

    # uses motion_detection_maske to make mask in the desired zone. 
    masked_zone = motion_detection_mask(mask_thresh, zone)

    # 3.  Closing: connects nearby foreground regions
    mask_merged = cv2.morphologyEx(masked_zone, cv2.MORPH_CLOSE, kernel,)
    
    # 4. Find contours
    contours, _ = cv2.findContours(mask_merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_contour_area = min_area # minimum area treshold
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    #frame_out = frame.copy()
    #for cnt in large_contours:
    #    x, y, w, h = cv2.boundingRect(cnt)
    #    frame_out = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 2)
    
    box = []
    for cnt in large_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            box.append((x,y,w,h))
     
    # Look at black white movement       
    #cv2.imshow("fg_mask raw", fg_mask)  #before shadow is removed
    cv2.imshow("Mask merged", mask_merged)
    return box


    



