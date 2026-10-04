
    # see the two lanes from our built in functions
    # create a diagonal waypoint to some random way point along middle of lane to left/right

    # change langes


import cv2
import os
from ultralytics import YOLO
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from hsv import hsv
from enum import Enum

class lane_changer_state(Enum):
    FORWARD = 1
    CHANGE_LANE = 2
    GO_TO_END_BARREL = 3



class lane_changer:
    def __init__(self):
        self.yellow_mask = None
        self.white_mask = None
        self.final = None
        self.hsv_obj = hsv('/home/potoo/Coding-Spaces/arv/UMARV-CV-ScenePerception/algorithms/algorithm_32rr2m7i/src/data/pedestrian_comp.mp4')
        self.cap = cv2.VideoCapture("/home/potoo/Coding-Spaces/arv/UMARV-CV-ScenePerception/algorithms/algorithm_32rr2m7i/src/data/pedestrian_comp.mp4")  
        self.THE_model = YOLO('./data/yolov8n.pt')
        self.frame_count = 0
        self.process_per_frame = 3
        self.has_changed_lane = False
        self.changing_lanes = False
        self.img = None
    # default YOLO model already has person as a class
    

    

    def update_masks(self):
        self.final, dict = self.hsv_obj.get_mask(self.img, yolo_barrels=True)
        self.white_mask = dict["white"]
        # self.yellow_mask = dict["yellow"]

    def find_middle_line(self):
        # make a waypoint at the middle dashed line
        x_middle = 100
        y_middle = 100

        cv2.circle(self.img, center = (x_middle,y_middle), radius = 10, color = (0,0,255), thickness = 1)
        return self.img

    def find_left_middle_line():
        return

    def change_lane(self, img):
        self.hsv_obj = hsv('/home/potoo/Coding-Spaces/arv/UMARV-CV-ScenePerception/algorithms/algorithm_32rr2m7i/src/data/pedestrian_comp.mp4') # for thresholds in hsv_values
            
        # while self.cap.isOpened():
        #     ret, self.image = self.cap.read()
        #     if ret:
        #         self.height, self.width, _ = self.image.shape
                    
        #         self.update_mask()
        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break
        #     else:
        #             break
        #     self.cap.release()
        #     cv2.destroyAllWindows()
        self.find_middle_line()
        # self.find_left_middle_line()
        return

    def run(self):
        # add enum for state machine
        # state 1 : lane keeping (checking for pedestrian in middle of screen)
        # state 2: lane changing
        # state 3: stopping
        while self.cap.isOpened():
            ret, self.img = self.cap.read()
            if ret:
                self.height, self.width, _ = self.img.shape
                self.update_masks()
            if not ret:
                break
            self.frame_count += 1
            if self.frame_count % self.process_per_frame != 0:
                continue
            if (self.changing_lanes):
                print()
            else:
                # run image through model
                
                results = self.THE_model(self.img)
                for results in results:
                    boxes = results.boxes.xyxy.tolist()
                    confidences = results.boxes.conf.tolist()
                    class_ids = results.boxes.cls.tolist()

                    for box, confidence, class_id in zip(boxes, confidences, class_ids):
                        # class 0 is person (built in) and adjust confidence as needed
                        # We see a person and we should show then lane change
                        if class_id == 0 and confidence > 0.7:
                            px1, py1, px2, py2 = map(int, box)

                            cv2.rectangle(self.img, (px1, py1), (px2, py2), (0, 255, 0), 2)
                            cv2.putText(self.img, f"Person", (px1, py1-10), 
                                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                            
                            if not self.has_changed_lane:
                                self.change_lane(self.img)
                                self.has_changed_lane = True
                            self.changing_lanes = True
                        
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    lane_changing = lane_changer()
    lane_changing.run()

if __name__ == "__main__":
    main()
    
    