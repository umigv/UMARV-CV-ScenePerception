import cv2
import numpy as np


class hsv:
    def __init__(self):
        self.image_path = 'data/curved_self_drive.jpg'
        self.image = cv2.imread(self.image_path)
        self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.hsv_value = None
        self.h_upper = None
        self.h_lower = None
        self.s_upper = None
        self.s_lower = None
        self.v_upper = None
        self.v_lower = None
        self.post_process = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.hsv_value = self.hsv_image[y, x]
            print('HSV value at ({}, {}): {}'.format(x, y, self.hsv_value))
            self.create_mask()

    def create_mask(self):
        self.h_lower = max(self.hsv_value[0] - 40, 0)
        self.h_upper = min(self.hsv_value[0] + 40, 179)
        self.s_upper = 255
        self.s_lower = 0
        self.v_upper = 255
        self.v_lower = 0
        lower_bound = np.array([self.h_lower, self.s_lower, self.v_lower])
        upper_bound = np.array([self.h_upper, self.s_upper, self.v_upper])
        mask = cv2.inRange(self.hsv_image, lower_bound, upper_bound)
        if self.post_process:
            cv2.erode(mask, None, iterations=2)
            cv2.dilate(mask, None, iterations=2)
        cv2.imshow('mask', mask)
        cv2.waitKey(0)

    def h_upper_callback(self, value):
        self.h_upper = value
        if self.h_upper < self.h_lower:
            self.h_upper = self.h_lower
        self.update_mask()

    def h_lower_callback(self, value):
        self.h_lower = value
        if self.h_lower > self.h_upper:
            self.h_lower = self.h_upper
        self.update_mask()

    def s_upper_callback(self, value):
        self.s_upper = value
        if self.s_upper < self.s_lower:
            self.s_upper = self.s_lower
        self.update_mask()

    def s_lower_callback(self, value):
        self.s_lower = value
        if self.s_lower > self.s_upper:
            self.s_lower = self.s_upper
        self.update_mask()

    def v_upper_callback(self, value):
        self.v_upper = value
        if self.v_upper < self.v_lower:
            self.v_upper = self.v_lower
        self.update_mask()

    def v_lower_callback(self, value):
        self.v_lower = value
        if self.v_lower > self.v_upper:
            self.v_lower = self.v_upper
        self.update_mask()

    def update_mask(self):
        lower_bound = np.array([self.h_lower, self.s_lower, self.v_lower])
        upper_bound = np.array([self.h_upper, self.s_upper, self.v_upper])
        mask = cv2.inRange(self.hsv_image, lower_bound, upper_bound)
        if self.post_process:
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
        result = cv2.bitwise_and(self.image, self.image, mask=mask)
        cv2.imshow('mask', mask)
        cv2.imshow('result', result)
        
    def main(self):
        cv2.imshow('raw image', self.image)
        cv2.setMouseCallback('raw image', self.mouse_callback)
        print("Click to show trackbars")
        cv2.waitKey(0)
        cv2.namedWindow('control pannel')
        cv2.createTrackbar('H_upper', 'control pannel', self.h_upper, 179, self.h_upper_callback)
        cv2.createTrackbar('H_lower', 'control pannel', self.h_lower, 179, self.h_lower_callback)
        cv2.createTrackbar('S_upper', 'control pannel', self.s_upper, 255, self.s_upper_callback)
        cv2.createTrackbar('S_lower', 'control pannel', self.s_lower, 255, self.s_lower_callback)
        cv2.createTrackbar('V_upper', 'control pannel', self.v_upper, 255, self.v_upper_callback)
        cv2.createTrackbar('V_lower', 'control pannel', self.v_lower, 255, self.v_lower_callback)
        
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()



if __name__ == '__main__':
    hsv = hsv()
    hsv.main()