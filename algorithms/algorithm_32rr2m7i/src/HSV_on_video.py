import cv2
import numpy as np
from time import sleep


class hsv:
    def __init__(self):
        self.video_path = 'data/IMG_4880.MOV'
        self.image = None
        self.mask = None
        self.hsv_image = None
        self.hsv_value = None
        self.contoured = None
        self.h_upper = 255
        self.h_lower = 0
        self.s_upper = 50
        self.s_lower = 0
        self.v_upper = 255
        self.v_lower = 236
        self.setup = False
        self.post_process = True
        
    def increase_contrast(self):
        self.image = cv2.convertScaleAbs(self.image, alpha=self.alpha, beta=self.beta)
            
    def find_line_cols(self, row, mask):
        line1 = 0
        in_a_row = 0
        for col in range(mask.shape[1]):
            if (mask[row, col] > 0):
                in_a_row += 1
                if(in_a_row == 5):
                    line1 = col
        line2 = 0
        in_a_row = 0
        for col in range(line1 + 30, mask.shape[1]):
            if (mask[row, col] > 0):
                in_a_row += 1
                if(in_a_row == 5):
                    line2 = col
        return line1, line2
                
    
    def find_closest_row(self, mask):
        # Iterate from the bottom to the top
        in_a_row = 0
        for row in range(mask.shape[0] - 1, -1, -1):
            if np.any(mask[row, :] > 0):
                in_a_row += 1
                if(in_a_row == 7):
                    return row-10
            else:
                in_a_row = 0
        return None
    
    def adjust_gamma(self, gamma=0.5):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        self.image = cv2.LUT(self.image, table)

    def h_upper_callback(self, value):
        self.h_upper = value
        if self.h_upper < self.h_lower:
            self.h_upper = self.h_lower
        self.update_mask()
        display = cv2.hconcat([self.image, cv2.cvtColor(self.contoured, cv2.COLOR_GRAY2BGR)])
        cv2.imshow('side by side', display)

    def h_lower_callback(self, value):
        self.h_lower = value
        if self.h_lower > self.h_upper:
            self.h_lower = self.h_upper
        self.update_mask()
        display = cv2.hconcat([self.image, cv2.cvtColor(self.contoured, cv2.COLOR_GRAY2BGR)])
        cv2.imshow('side by side', display)

    def s_upper_callback(self, value):
        self.s_upper = value
        if self.s_upper < self.s_lower:
            self.s_upper = self.s_lower
        self.update_mask()
        display = cv2.hconcat([self.image, cv2.cvtColor(self.contoured, cv2.COLOR_GRAY2BGR)])
        cv2.imshow('side by side', display)

    def s_lower_callback(self, value):
        self.s_lower = value
        if self.s_lower > self.s_upper:
            self.s_lower = self.s_upper
        self.update_mask()
        display = cv2.hconcat([self.image, cv2.cvtColor(self.contoured, cv2.COLOR_GRAY2BGR)])
        cv2.imshow('side by side', display)

    def v_upper_callback(self, value):
        self.v_upper = value
        if self.v_upper < self.v_lower:
            self.v_upper = self.v_lower
        self.update_mask()
        display = cv2.hconcat([self.image, cv2.cvtColor(self.contoured, cv2.COLOR_GRAY2BGR)])
        cv2.imshow('side by side', display)

    def v_lower_callback(self, value):
        self.v_lower = value
        if self.v_lower > self.v_upper:
            self.v_lower = self.v_upper
        self.update_mask()
        display = cv2.hconcat([self.image, cv2.cvtColor(self.contoured, cv2.COLOR_GRAY2BGR)])
        cv2.imshow('side by side', display)

    def update_mask(self):
        lower_bound = np.array([self.h_lower, self.s_lower, self.v_lower])
        upper_bound = np.array([self.h_upper, self.s_upper, self.v_upper])
        self.mask = cv2.inRange(self.hsv_image, lower_bound, upper_bound)
        if self.post_process:
            self.mask = cv2.erode(self.mask, None, iterations=2)
            self.mask = cv2.dilate(self.mask, None, iterations=2)
            contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 1500 # Adjust based on noise size
            self.contoured = np.zeros_like(self.mask)
            for cnt in contours:
                if cv2.contourArea(cnt) > min_area:
                    cv2.drawContours(self.contoured, [cnt], -1, 255, thickness=cv2.FILLED)

        # Find the closest row from the bottom where the lane lines start
        # closest_row = self.find_closest_row(mask)
        # if closest_row is not None:
        #     l1, l2 = self.find_line_cols(closest_row, mask)
        #     print(f"Closest row from the bottom where lane lines start: {closest_row}")
        #     cv2.line(result, (0, closest_row), (result.shape[1], closest_row), (0, 255, 0), 2)
        #     cv2.line(result, (l1, closest_row), (0, result.shape[0]), (255, 0, 0), 2)  # Line to bottom-left corner
        #     cv2.line(result, (l2, closest_row), (result.shape[1], result.shape[0]), (255, 0, 0), 2)  # Line to bottom-right corner
        #     cv2.line(mask, (l1, closest_row), (0, result.shape[0]), (255, 0, 0), 10)  # Line to bottom-left corner
        #     cv2.line(mask, (l2, closest_row), (result.shape[1], result.shape[0]), (255, 0, 0), 10)  # Line to bottom-right corner
        #     cv2.imshow('result_with_line', result)
        #     cv2.imshow('mask', mask)
        
    def alpha_callback(self, value):
        self.alpha = value / 100.0  # Map the integer value to the range 1.0 to 3.0
        self.increase_contrast()
        display = cv2.hconcat([self.image, cv2.cvtColor(self.contoured, cv2.COLOR_GRAY2BGR)])
        cv2.imshow('side by side', display)
        
    def beta_callback(self, value):
        self.beta = value
        self.increase_contrast()
        display = cv2.hconcat([self.image, cv2.cvtColor(self.contoured, cv2.COLOR_GRAY2BGR)])
        cv2.imshow('side by side', display)
        
    def on_button_click(self, value):
        if(value == 1):
            self.setup = not self.setup
        
    def main(self):
        self.setup = True
        cap = cv2.VideoCapture(self.video_path)
        cv2.namedWindow('raw video')
        
        cv2.namedWindow('control pannel')
        cv2.createTrackbar('H_upper', 'control pannel', self.h_upper, 179, self.h_upper_callback)
        cv2.createTrackbar('H_lower', 'control pannel', self.h_lower, 179, self.h_lower_callback)
        cv2.createTrackbar('S_upper', 'control pannel', self.s_upper, 255, self.s_upper_callback)
        cv2.createTrackbar('S_lower', 'control pannel', self.s_lower, 255, self.s_lower_callback)
        cv2.createTrackbar('V_upper', 'control pannel', self.v_upper, 255, self.v_upper_callback)
        cv2.createTrackbar('V_lower', 'control pannel', self.v_lower, 255, self.v_lower_callback)
        cv2.createTrackbar('Done Filtering', 'control pannel', 0, 1, self.on_button_click)


        while self.setup == True:
            ret, self.image = cap.read()
            if ret:
                self.adjust_gamma()
                self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
                self.update_mask()
                #concatonate self.mask and self.image
                
                display = cv2.hconcat([self.image, cv2.cvtColor(self.contoured, cv2.COLOR_GRAY2BGR)])
                cv2.imshow('side by side', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                sleep(0.01)
            else:
                cap.release()
                cap = cv2.VideoCapture(self.video_path)
                break
        
        cap.release()
        cv2.destroyWindow('control pannel')
        cv2.destroyWindow('side by side')
        
        sleep(3)
            
        
        cap2 = cv2.VideoCapture(self.video_path)
        

        while self.setup == False:
            ret, self.image = cap2.read()
            if ret:
                self.adjust_gamma()
                self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
                self.update_mask()
                cv2.imshow('Lane Lines mask', self.contoured)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        
        cv2.destroyAllWindows()
        
        



if __name__ == '__main__':
    hsv = hsv()
    hsv.main()