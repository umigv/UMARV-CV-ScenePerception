import cv2
import numpy as np
import os
import json
from ultralytics import YOLO

class hsv:
    def __init__(self, video_path):
        self.hsv_image = None
        self.hsv_filters = {}  # Map of filter names to HSV bounds
        self.setup = False
        self.image = None
        self.final = None
        self.barrel = False
        self.video_path = video_path
        self.barrel_mask = None
        self.barrel_boxes = None
        self.barrel_model =  YOLO("data/obstacles.pt")
        self.lane_model = YOLO("data/laneswithcontrast.pt")
        self.load_hsv_values()
        
        
    def load_hsv_values(self):
        if os.path.exists('hsv_values.json'):
            with open('hsv_values.json', 'r') as file:
                all_hsv_values = json.load(file)
                self.hsv_filters = all_hsv_values.get(str(self.video_path), {})
        else:
            # print("Matt put it in the wrong spot")
            # Initialize with an empty filter map if the JSON file doesn't exist
            self.hsv_filters["white"] = {
                'h_upper': 29, 'h_lower': 0,
                's_upper': 51, 's_lower': 0,
                'v_upper': 255, 'v_lower': 137
            }
            # print(self.hsv_filters)

    def save_hsv_values(self):
        all_hsv_values = {}
        if os.path.exists('hsv_values.json'):
            with open('hsv_values.json', 'r') as file:
                all_hsv_values = json.load(file)
        all_hsv_values[str(self.video_path)] = self.hsv_filters
        with open('hsv_values.json', 'w') as file:
            json.dump(all_hsv_values, file, indent=4)

    def h_upper_callback(self, value):
        self.h_upper = value
        if self.h_upper < self.h_lower:
            self.h_upper = self.h_lower
        self.update_mask()
        cv2.imshow("Mask", self.final)

    def h_lower_callback(self, value):
        self.h_lower = value
        if self.h_lower > self.h_upper:
            self.h_lower = self.h_upper
        self.update_mask()
        cv2.imshow("Mask", self.final)

    def s_upper_callback(self, value):
        self.s_upper = value
        if self.s_upper < self.s_lower:
            self.s_upper = self.s_lower
        self.update_mask()
        cv2.imshow("Mask", self.final)

    def s_lower_callback(self, value):
        self.s_lower = value
        if self.s_lower > self.s_upper:
            self.s_lower = self.s_upper
        self.update_mask()
        cv2.imshow("Mask", self.final)

    def v_upper_callback(self, value):
        self.v_upper = value
        if self.v_upper < self.v_lower:
            self.v_upper = self.v_lower
        self.update_mask()
        cv2.imshow("Mask", self.final)

    def v_lower_callback(self, value):
        self.v_lower = value
        if self.v_lower > self.v_upper:
            self.v_lower = self.v_upper
        self.update_mask()
        cv2.imshow("Mask", self.final)
        
    def on_button_click(self, value):
        if(value == 1):
            self.setup = False
            
    def __update_filter(self, filter_name, key, value):
        self.hsv_filters[filter_name][key] = value
        _, filters = self.update_mask()
        cv2.imshow("Mask", filters[filter_name])

    def clear_filter(self, filter_name):
        if os.path.exists('hsv_values.json'):
            with open('hsv_values.json', 'r') as file:
                all_hsv_values = json.load(file)

            if self.video_path in all_hsv_values:
                if filter_name in all_hsv_values[self.video_path]:
                    del all_hsv_values[self.video_path][filter_name]

                    if not all_hsv_values[self.video_path]:
                        del all_hsv_values[self.video_path]

                    with open('hsv_values.json', 'w') as file:
                        json.dump(all_hsv_values, file, indent=4)
                    print(f"Filter '{filter_name}' cleared for video '{self.video_path}'.")
                else:
                    print(f"Filter '{filter_name}' does not exist for video '{self.video_path}'.")
            else:
                print(f"Video '{self.video_path}' does not exist in the JSON file.")
        else:
            print("No HSV values file found.")
                
    def get_barrels_YOLO(self):
        # Get the driveable area of one frame and return the inverted mask
        results = self.barrel_model.predict(self.image, conf=0.7)[0]
        self.barrel_mask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
        if results.boxes is not None:
            self.barrel_boxes = results.boxes.xyxyn
        else:
            self.barrel_boxes = None
        if(results.masks is not None):
            for i in range(len(results.masks.xy)):
                    segment = results.masks.xy[i]
                    segment_array = np.array([segment], dtype=np.int32)
                    cv2.fillPoly(self.barrel_mask, [segment_array], color=(255, 0, 0))
        return self.barrel_mask
    
    def adjust_gamma(self, gamma=0.4):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        self.image = cv2.LUT(self.image, table)
        
    def tune(self, filter_name):
        if filter_name not in self.hsv_filters:
            # Initialize default values for the new filter
            self.hsv_filters[filter_name] = {
                'h_upper': 179, 'h_lower': 0,
                's_upper': 255, 's_lower': 0,
                'v_upper': 255, 'v_lower': 0
            }
        filter_values = self.hsv_filters[filter_name]
        self.setup = True
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file {self.video_path}")
            return

        cv2.namedWindow('control pannel')
        cv2.createTrackbar('H_upper', 'control pannel', filter_values['h_upper'], 179,
                           lambda v: self.__update_filter(filter_name, 'h_upper', v))
        cv2.createTrackbar('H_lower', 'control pannel', filter_values['h_lower'], 179,
                           lambda v: self.__update_filter(filter_name, 'h_lower', v))
        cv2.createTrackbar('S_upper', 'control pannel', filter_values['s_upper'], 255,
                           lambda v: self.__update_filter(filter_name, 's_upper', v))
        cv2.createTrackbar('S_lower', 'control pannel', filter_values['s_lower'], 255,
                           lambda v: self.__update_filter(filter_name, 's_lower', v))
        cv2.createTrackbar('V_upper', 'control pannel', filter_values['v_upper'], 255,
                           lambda v: self.__update_filter(filter_name, 'v_upper', v))
        cv2.createTrackbar('V_lower', 'control pannel', filter_values['v_lower'], 255,
                           lambda v: self.__update_filter(filter_name, 'v_lower', v))
        cv2.createTrackbar('Done Tuning', 'control pannel', 0, 1, self.on_button_click)

        while self.setup:
            ret, frame = cap.read()
            if not ret:
                # If the video ends, reset to the beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            self.image = frame
            self.adjust_gamma()
            self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            mask, dict = self.update_mask()

            cv2.imshow('Video', frame)
            cv2.imshow('Mask', dict[filter_name])

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Press 'Esc' to exit the loop
                break

        cap.release()
        cv2.destroyAllWindows()
        self.save_hsv_values()

    def get_lane_lines_YOLO(self):
        # Get the driveable area of one frame and return the inverted mask
        results = self.lane_model.predict(self.image, conf=0.7)[0]
        laneline_mask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
        if(results.masks is not None):
            for i in range(len(results.masks.xy)):
                    segment = results.masks.xy[i]
                    segment_array = np.array([segment], dtype=np.int32)
                    cv2.fillPoly(laneline_mask, [segment_array], color=(255, 0, 0))
        return laneline_mask
        
    def update_mask(self):
        combined_mask = None
        masks = {}

        for filter_name, bounds in self.hsv_filters.items():
            lower_bound = np.array([bounds["h_lower"], bounds['s_lower'], bounds['v_lower']])
            upper_bound = np.array([bounds['h_upper'], bounds['s_upper'], bounds['v_upper']])
            mask = cv2.inRange(self.hsv_image, lower_bound, upper_bound)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=4)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 200 # Adjust based on noise size
            final = np.zeros_like(mask)
            for cnt in contours:
                if cv2.contourArea(cnt) > min_area:
                    cv2.drawContours(final, [cnt], -1, 255, thickness=cv2.FILLED)

            if filter_name == "white" and self.YOLO_lanes:
                lane_line_mask = self.get_lane_lines_YOLO()
                final = cv2.bitwise_or(final, lane_line_mask)
            # Combine masks
            if combined_mask is None:
                combined_mask = final
            else:
                combined_mask = cv2.bitwise_or(combined_mask, final)

            masks[filter_name] = final

        if self.YOLO_barrels:
            barrels = self.get_barrels_YOLO()
            combined_mask = cv2.bitwise_or(combined_mask, barrels)

        return combined_mask, masks
        
    def get_mask(self, frame, yolo_lanes=False, yolo_barrels=False):
        self.YOLO_lanes = yolo_lanes
        self.YOLO_barrels = yolo_barrels
        self.image = frame
        self.adjust_gamma()
        self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        return self.update_mask()
