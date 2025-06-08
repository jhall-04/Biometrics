import cv2

cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Camera is accessible")
else:
    print("Camera is NOT accessible")
cap.release()