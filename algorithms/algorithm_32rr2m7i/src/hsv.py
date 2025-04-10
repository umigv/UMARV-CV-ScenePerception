import cv2
import numpy as np
import os
import json

class hsv:
    def __init__(self, video_path):
        self.hsv_image = None
        self.hsv_filters = {}  # Map of filter names to HSV bounds
        self.setup = False
        self.image = None
        self.final = None
        self.video_path = video_path
        self.load_hsv_values()
        
    def load_hsv_values(self):
        if os.path.exists('hsv_values.json'):
            with open('hsv_values.json', 'r') as file:
                all_hsv_values = json.load(file)
                self.hsv_filters = all_hsv_values.get(self.video_path, {})
        else:
            # Initialize with an empty filter map if the JSON file doesn't exist
            self.hsv_filters = {}

    def save_hsv_values(self):
        all_hsv_values = {}
        if os.path.exists('hsv_values.json'):
            with open('hsv_values.json', 'r') as file:
                all_hsv_values = json.load(file)
        all_hsv_values[self.video_path] = self.hsv_filters
        with open('hsv_values.json', 'w') as file:
            json.dump(all_hsv_values, file, indent=4)
    # def load_hsv_values(self):
    #     if os.path.exists('hsv_values.json'):
    #         with open('hsv_values.json', 'r') as file:
    #             all_hsv_values = json.load(file)
    #             video_hsv_values = all_hsv_values.get(self.video_path, {})
    #             self.h_upper = video_hsv_values.get('h_upper', 179)  # Default value
    #             self.h_lower = video_hsv_values.get('h_lower', 0)    # Default value
    #             self.s_upper = video_hsv_values.get('s_upper', 255)  # Default value
    #             self.s_lower = video_hsv_values.get('s_lower', 0)    # Default value
    #             self.v_upper = video_hsv_values.get('v_upper', 255)  # Default value
    #             self.v_lower = video_hsv_values.get('v_lower', 0)    # Default value
    #     else:
    #         # Set default values if the JSON file doesn't exist
    #         self.h_upper = 179
    #         self.h_lower = 0
    #         self.s_upper = 255
    #         self.s_lower = 0
    #         self.v_upper = 255
    #         self.v_lower = 0
                
    # def save_hsv_values(self):
    #     all_hsv_values = {}
    #     if os.path.exists('hsv_values.json'):
    #         with open('hsv_values.json', 'r') as file:
    #             all_hsv_values = json.load(file)
    #     all_hsv_values[self.video_path] = {
    #         'h_upper': self.h_upper,
    #         'h_lower': self.h_lower,
    #         's_upper': self.s_upper,
    #         's_lower': self.s_lower,
    #         'v_upper': self.v_upper,
    #         'v_lower': self.v_lower
    #     }
    #     with open('hsv_values.json', 'w') as file:
    #         json.dump(all_hsv_values, file, indent=4)


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

    # def update_mask(self):
    #     lower_bound = np.array([self.h_lower, self.s_lower, self.v_lower])
    #     upper_bound = np.array([self.h_upper, self.s_upper, self.v_upper])
    #     self.mask = cv2.inRange(self.hsv_image, lower_bound, upper_bound)
    #     self.mask = cv2.erode(self.mask, None, iterations=2)
    #     self.mask = cv2.dilate(self.mask, None, iterations=2)
    #     contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     min_area = 200 # Adjust based on noise size
    #     self.final = np.zeros_like(self.mask)
    #     for cnt in contours:
    #         if cv2.contourArea(cnt) > min_area:
    #             cv2.drawContours(self.final, [cnt], -1, 255, thickness=cv2.FILLED)
                
    #     return self.final
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

            # Combine masks
            if combined_mask is None:
                combined_mask = final
            else:
                combined_mask = cv2.bitwise_or(combined_mask, final)

            masks[filter_name] = final

        return combined_mask, masks
        
    def get_mask(self, frame):
        self.image = frame
        self.adjust_gamma()
        self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        return self.update_mask()
