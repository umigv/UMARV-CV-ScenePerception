import cv2
from ultralytics import YOLO
import numpy as np
import sys
import matplotlib.pyplot as plt
import json
import time
import math


# def num_difference(x, y, total):
#     if y != 0 and x == 0:
#         total += 1
#     return total

def predict(video_path, lane_model, hole_model=None):
    total = 0
    lane_model = YOLO(lane_model) # specify the model you would like to use
    hole_model = YOLO(hole_model) if hole_model is not None else None

    video_path = video_path #make 0 if using webcam
    cap = cv2.VideoCapture(video_path)
    
    # cap.set(cv2.CAP_PROP_MODE, cv2.CAP_MODE_GRAY)
    
    image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    occupancy_grid = np.zeros((image_height, image_width))
    memory_buffer = np.full((image_height, image_width), 255).astype(np.uint8) #stack with past frames of driveable area (intialize with full)
    # print(memory_buffer)
    time_of_buffer = 0
    buffer_area = 0
    # count = 0
    cur_time = time.time()
    while cap.isOpened() and cv2.waitKey(1) & 0xFF != ord("q"):
        success, frame = cap.read()
        
        if success:
            r_lane = lane_model.predict(frame, conf=0.7)[0] # this makes a prediction on a single frame of video
            r_hole = hole_model.predict(frame, conf=0.25)[0] if hole_model is not None else None
            lane_annotated_frame = r_lane.plot() # 
            hole_annotated_frame = r_hole.plot() if hole_model is not None else None
            
            # image_height = frame.shape[0]
            # image_width = frame.shape[1]
            occupancy_grid = np.zeros((image_height, image_width))
            # print(len(r_lane.masks.xy))
            
            
            if r_lane.masks is not None:
                if(len(r_lane.masks.xy) != 0):
                    for segment in r_lane.masks.xy:
                        segment_array = np.array([segment], dtype=np.int32)
                        cv2.fillPoly(occupancy_grid, [segment_array], color=(255, 0, 0))
                        current_time = time.time()
                        difference = current_time - time_of_buffer
                        print(np.logical_not(occupancy_grid, np.logical_not(memory_buffer)))
                        switch = np.sum(np.logical_and(memory_buffer, np.logical_not(occupancy_grid)))/(np.sum(memory_buffer)/255)
                        print("DIFFERENCE", difference)  
                        if switch >= 0.8 and difference < 4:
                            occupancy_grid = memory_buffer
                            print("FLIPPED")
                            cv2.waitKey(3000)
                        elif switch >= 0.8 and difference < 8:
                            occupancy_grid.fill(255)
                            cv2.waitKey(3000)
                        else:
                            memory_buffer = occupancy_grid # add the most recent grid as a memory buffer
                            time_of_buffer = time.time()

                        buffer_area = np.sum(occupancy_grid)//255
                        #update the frame when the most recent buffer was gathered
                        
                        for i in range(occupancy_grid.shape[1]):
                            if np.any(occupancy_grid[-200:, i]):
                                occupancy_grid[-50:, i] = 255
                    
                    
            else:
                # if no detections are made we can use past detections or a fully filled grid as output
                # 
                current_time = time.time()
                buffer_time = math.exp(-buffer_area/(image_width*image_height)-0.7)# between 1 and 1/e 
                if current_time - time_of_buffer < buffer_time: 
                    #number 10 can be changed if needed, this is the number of frames between the buffer and the current frame for it to be relevent
                    occupancy_grid = memory_buffer
                    # print(occupancy_grid)
                    print("BUFFER USED")
                    cv2.waitKey(3000)
                else:
                    occupancy_grid.fill(255)
                    print("FULL OCCUPANCY GRID USED")
                    cv2.waitKey(3000)
                    

            '''
            This can be added to fill occupancy grid when no detections are made
            else: 
                occupancy_grid.fill(255)
            '''
            if r_hole is not None:
                if r_hole.boxes is not None:
                    for segment in r_hole.boxes.xyxyn:
                        x_min, y_min, x_max, y_max = segment
                        vertices = np.array([[x_min*image_width, y_min*image_height], 
                                            [x_max*image_width, y_min*image_height], 
                                            [x_max*image_width, y_max*image_height], 
                                            [x_min*image_width, y_max*image_height]], dtype=np.int32)
                        cv2.fillPoly(occupancy_grid, [vertices], color=(0, 0, 0))

            occupancy_grid = cv2.resize(occupancy_grid, (image_height//2, image_width//2))
            frame = cv2.resize(frame, (image_height//2,image_width//2))
            lane_annotated_frame = cv2.resize(lane_annotated_frame, (image_height//2,image_width//2))

            cv2.imshow("Lane Lines", occupancy_grid)
            cv2.imshow("frame", frame)
            cv2.imshow("YOLOv8 Inference", lane_annotated_frame)


            # if(len(r_lane.masks.xy) > 1):
            #     print(r_lane.boxes.conf[0])
            #     cv2.waitKey(10000)
                
            
            
            ##################For Nav Output not necessary for running
            # summed_grid = np.sum(occupancy_grid, axis=2)

            # BINARY GRID TO SEND TO NAV
            # binary_grid = np.array(np.where(summed_grid == 0, 0, 1))
            # print("here5")
            #################################
            # plt.imshow(binary_grid, cmap='binary_r')
            # plt.show()
            # if hole_model is not None:
            #     cv2.imshow("Potholes", hole_annotated_frame)
            
            
            if cv2.waitKey(1) & 0xFF == ord("q"): # press q to quit the program
                break
        else:
            break
        
    ##########################
    # Must have these dont touch
    ##########################
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Check if the script is run as the main program
    if len(sys.argv) < 3:
        print("\nNot enough parameters!! Please enter 1 of the 3:\n ")
        print("1) python3 yolov8.py <video_path> <lane_line_model_path> <pot_holes_model_path>")
        print("2) python3 yolov8.py <video_path> <lane_line_model_path>")
        print("3) enter 0 for the <video_path> if you are using a webcam device\n")
        sys.exit(1)

    # Extract the command line argument (parameter)
    parameter_value = sys.argv[1]
    model_name = sys.argv[2]
    hole_model = None
    
    
    if len(sys.argv) == 4:
        hole_model = sys.argv[3]
    if(parameter_value == "0"):
        parameter_value = int(parameter_value)

    # Call your function with the provided parameter
    predict(parameter_value, model_name, hole_model)