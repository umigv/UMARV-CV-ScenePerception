import cv2
import torch
import numpy as np
from ultralytics import YOLO

model = YOLO('./data/tire.pt')

def run_tire_test():
    cap = cv2.VideoCapture(0) # add data path here if testing on video
    test_started = False

    print("Test ready.")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        display_frame = frame.copy()

        

        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            test_started = True
            print("Test Started")
        elif key == ord('q'):
            break

    cap.release()

if __name__ == "__main__":
    run_tire_test()

