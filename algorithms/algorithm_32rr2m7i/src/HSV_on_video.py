import cv2
import numpy as np
from time import sleep
from ultralytics import YOLO


class hsv:
    def __init__(self):
        self.video_path = 'data/right_turn.mp4'
        self.image = None
        self.rgb_image = None
        self.mask = None
        self.hsv_image = None
        self.hsv_value = None
        self.contoured = None
        self.image_height = None
        self.image_width = None
        self.h_upper = 255
        self.h_lower = 0
        self.s_upper = 50
        self.s_lower = 0
        self.v_upper = 255
        self.v_lower = 236
        self.setup = True
        self.post_process = True
        self.nomansland = True
        self.laneline_mask = None
        self.driveable_mask = None
        self.occupancy_grid = None
        self.lanes = None
        # self.laneline_model = YOLO('data/laneswithcontrast.pt')
        # self.model = YOLO('data/bestDriveableArea.pt')
    
    def find_line_cols(self, row):
        line1 = 0
        in_a_row = 0
        for col in range(self.occupancy_grid.shape[1]):
            if (self.occupancy_grid[row, col] > 0):
                in_a_row += 1
                if(in_a_row == 5):
                    line1 = col
        line2 = self.image_width-10
        in_a_row = 0
        for col in range(line1 + 100, self.occupancy_grid.shape[1]):
            if (self.occupancy_grid[row, col] > 0):
                in_a_row += 1
                if(in_a_row == 5):
                    line2 = col
        return line1, line2
                
    
    def find_closest_row(self):
        # Iterate from the bottom to the top
        in_a_row = 0
        for row in range(self.occupancy_grid.shape[0] - 1, -1, -1):
            if np.any(self.occupancy_grid[row, :] > 0):
                in_a_row += 1
                if(in_a_row == 7):
                    return row-50
            else:
                in_a_row = 0
        return None
    
    def adjust_gamma(self, gamma=0.4):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        self.image = cv2.LUT(self.image, table)

    def h_upper_callback(self, value):
        self.h_upper = value
        if self.h_upper < self.h_lower:
            self.h_upper = self.h_lower
        self.update_mask()
        display = cv2.hconcat([self.image, cv2.cvtColor(self.occupancy_grid, cv2.COLOR_GRAY2BGR)])
        cv2.imshow('side by side', display)

    def h_lower_callback(self, value):
        self.h_lower = value
        if self.h_lower > self.h_upper:
            self.h_lower = self.h_upper
        self.update_mask()
        display = cv2.hconcat([self.image, cv2.cvtColor(self.occupancy_grid, cv2.COLOR_GRAY2BGR)])
        cv2.imshow('side by side', display)

    def s_upper_callback(self, value):
        self.s_upper = value
        if self.s_upper < self.s_lower:
            self.s_upper = self.s_lower
        self.update_mask()
        display = cv2.hconcat([self.image, cv2.cvtColor(self.occupancy_grid, cv2.COLOR_GRAY2BGR)])
        cv2.imshow('side by side', display)

    def s_lower_callback(self, value):
        self.s_lower = value
        if self.s_lower > self.s_upper:
            self.s_lower = self.s_upper
        self.update_mask()
        display = cv2.hconcat([self.image, cv2.cvtColor(self.occupancy_grid, cv2.COLOR_GRAY2BGR)])
        cv2.imshow('side by side', display)

    def v_upper_callback(self, value):
        self.v_upper = value
        if self.v_upper < self.v_lower:
            self.v_upper = self.v_lower
        self.update_mask()
        display = cv2.hconcat([self.image, cv2.cvtColor(self.occupancy_grid, cv2.COLOR_GRAY2BGR)])
        cv2.imshow('side by side', display)

    def v_lower_callback(self, value):
        self.v_lower = value
        if self.v_lower > self.v_upper:
            self.v_lower = self.v_upper
        self.update_mask()
        display = cv2.hconcat([self.image, cv2.cvtColor(self.occupancy_grid, cv2.COLOR_GRAY2BGR)])
        cv2.imshow('side by side', display)
        
    def get_lane_lines_YOLO(self):
        # Get the driveable area of one frame and return the inverted mask
        results = self.laneline_model.predict(self.rgb_image, conf=0.7)[0]
        self.laneline_mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        if(results.masks is not None):
            segment = results.masks.xy[0]
            segment_array = np.array([segment], dtype=np.int32)
            cv2.fillPoly(self.laneline_mask, [segment_array], color=(255, 0, 0))
        
    def get_lane_line_mask(self):
        lower_bound = np.array([self.h_lower, self.s_lower, self.v_lower])
        upper_bound = np.array([self.h_upper, self.s_upper, self.v_upper])
        self.mask = cv2.inRange(self.hsv_image, lower_bound, upper_bound)
        self.mask = cv2.erode(self.mask, None, iterations=2)
        self.mask = cv2.dilate(self.mask, None, iterations=2)
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 200 # Adjust based on noise size
        hsv_lanes = np.zeros_like(self.mask)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(hsv_lanes, [cnt], -1, 255, thickness=cv2.FILLED)
        
        if not self.nomansland:
            # self.get_lane_lines_YOLO() # this updates laneline_mask
            self.lanes = cv2.bitwise_or(self.laneline_mask, hsv_lanes)
        else:
            self.lanes = hsv_lanes

    def update_mask(self):
        # get the inverted driveable area add it to self.contoured
        self.get_lane_line_mask()
        self.occupancy_grid = self.lanes
        if self.setup == False:
            if self.nomansland:
                closest_row = self.find_closest_row()
                if closest_row is not None:
                    l1, l2 = self.find_line_cols(closest_row)
                    # cv2.line(self.occupancy_grid, (0, closest_row), (self.occupancy_grid.shape[1], closest_row), (0, 255, 0), 2)
                    # cv2.line(self.occupancy_grid, (l1, closest_row), (0, self.occupancy_grid.shape[0]), (255, 0, 0), 2)  # Line to bottom-left corner
                    # cv2.line(self.occupancy_grid, (l2, closest_row), (self.occupancy_grid.shape[1], self.occupancy_grid.shape[0]), (255, 0, 0), 2)  # Line to bottom-right corner
                    cv2.line(self.occupancy_grid, (l1, closest_row), (0, self.occupancy_grid.shape[0]), (255, 0, 0), 10)  # Line to bottom-left corner
                    cv2.line(self.occupancy_grid, (l2, closest_row), (self.occupancy_grid.shape[1], self.occupancy_grid.shape[0]), (255, 0, 0), 10)  # Line to bottom-right corner
                self.get_barrels()
            else:
                self.get_inverted_driveable_area()
                # if driveable_area.shape != self.contoured.shape:
                #     driveable_area = cv2.resize(driveable_area, (self.contoured.shape[1], self.contoured.shape[0]))
                # driveable_area = driveable_area.astype(self.contoured.dtype)
                self.occupancy_grid = cv2.bitwise_or(self.lanes, self.driveable_mask)
                
        
    def on_button_click(self, value):
        if(value == 1):
            self.setup = False
        
    def setup_loop(self):
        self.setup = True
        cap = cv2.VideoCapture(self.video_path)
        self.image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cv2.namedWindow('control pannel')
        cv2.createTrackbar('H_upper', 'control pannel', self.h_upper, 179, self.h_upper_callback)
        cv2.createTrackbar('H_lower', 'control pannel', self.h_lower, 179, self.h_lower_callback)
        cv2.createTrackbar('S_upper', 'control pannel', self.s_upper, 255, self.s_upper_callback)
        cv2.createTrackbar('S_lower', 'control pannel', self.s_lower, 255, self.s_lower_callback)
        cv2.createTrackbar('V_upper', 'control pannel', self.v_upper, 255, self.v_upper_callback)
        cv2.createTrackbar('V_lower', 'control pannel', self.v_lower, 255, self.v_lower_callback)
        cv2.createTrackbar('Done Filtering', 'control pannel', 0, 1, self.on_button_click)
        while self.setup == True:
            print("setup loop")
            ret, self.image = cap.read()
            self.rgb_image = self.image
            if ret:
                # self.adjust_gamma()
                self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
                self.update_mask()
                #concatonate self.mask and self.image
                
                display = cv2.hconcat([self.image, cv2.cvtColor(self.occupancy_grid, cv2.COLOR_GRAY2BGR)])
                cv2.imshow('side by side', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                sleep(0.01)
            else:
                cap.release()
                cap = cv2.VideoCapture(self.video_path)
        
        cap.release()
        cv2.destroyWindow('control pannel')
        cv2.destroyWindow('side by side')
        
        sleep(2)
        self.setup = False
        self.main_loop()
        
    def get_barrels(self):
        lower_bound = np.array([0, 0, 111])
        upper_bound = np.array([179, 107, 255])
        barrels = cv2.inRange(self.hsv_image, lower_bound, upper_bound)
        barrels = cv2.bitwise_not(barrels)
        barrels = cv2.erode(barrels, None, iterations=2)
        barrels = cv2.dilate(barrels, None, iterations=2)
        self.occupancy_grid = cv2.bitwise_or(self.occupancy_grid, barrels)
        
    def get_inverted_driveable_area(self):
        # Get the driveable area of one frame and return the inverted mask
        results = self.model.predict(self.rgb_image, conf=0.7)[0]
        self.driveable_mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        if(results.masks is not None):
            segment = results.masks.xy[0]
            segment_array = np.array([segment], dtype=np.int32)
            cv2.fillPoly(self.driveable_mask, [segment_array], color=(255, 0, 0))
            self.driveable_mask = cv2.bitwise_not(self.driveable_mask)
        

    def main_loop(self):
        print("main loop starting")
        cap2 = cv2.VideoCapture(self.video_path)
        self.image_width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.image_height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.nomansland = True
        
        while self.setup == False:
            ret, self.image = cap2.read()
            self.rgb_image = self.image
            if ret:
                self.adjust_gamma()
                self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
                self.update_mask()
                
                
                cv2.imshow('raw image', self.image)
                cv2.imshow('Lane Lines mask', self.occupancy_grid)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        
        cv2.destroyAllWindows()


if __name__ == '__main__':
    hsv = hsv()
    hsv.setup_loop()