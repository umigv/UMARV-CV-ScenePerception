import cv2
from ultralytics import YOLO
import numpy as np
import sys
import time
import math

def predict(video_path, lane_model):
    lane_model = YOLO(lane_model) 
    cap = cv2.VideoCapture(video_path)
    
    image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while cap.isOpened() and cv2.waitKey(1) & 0xFF != ord("q"):
        success, frame = cap.read()
        
        if success:
            r_lane = lane_model.predict(frame, conf=0.7)[0] 
            occupancy_grid = np.zeros((image_height, image_width), dtype=np.uint8)
            
            if r_lane.masks is not None and len(r_lane.masks.xy) != 0:
                segment = r_lane.masks.xy[0]
                segment_array = np.array([segment], dtype=np.int32)
                cv2.fillPoly(occupancy_grid, [segment_array], color=(255, 255, 255)) 
            else:
                occupancy_grid.fill(255)  
            
            cv2.imwrite("occupancy_grid.png", occupancy_grid)
            
            frame_resized = cv2.resize(frame, (image_width // 2, image_height // 2))
            occupancy_resized = cv2.resize(occupancy_grid, (image_width // 2, image_height // 2))
            
            cv2.imshow("Frame", frame_resized)
            cv2.imshow("Occupancy Grid", occupancy_resized)
            
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\nNot enough parameters!! Please enter:\n")
        print("python3 yolov8.py <video_path> <lane_line_model_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    model_name = sys.argv[2]
    
    if video_path == "0":
        video_path = int(video_path)
    
    predict(video_path, model_name)
