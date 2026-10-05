import numpy as np
import cv2


# V

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
fps = cam.get(cv2.CAP_PROP_FPS)
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))


# Define the codec and create VideoWriter object

while True:
    ret, frame = cam.read()

    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()



