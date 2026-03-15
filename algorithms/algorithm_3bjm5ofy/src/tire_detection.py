import cv2
import torch
import numpy
from ultralytics import YOLO

model = YOLO('./data/tire.pt')

def run_tire_test():
    cap = cv2.VideoCapture(0) # add data path here if testing on video

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        display_frame = frame.copy()

