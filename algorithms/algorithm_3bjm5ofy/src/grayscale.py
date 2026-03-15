import cv2
import numpy as np


class grayscale:
    def __init__(self):
        self.image_path = 'data/000000.jpg'
        self.image = cv2.imread(self.image_path)
        self.grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.grayscale_value = None
        self.upper = None
        self.lower = None
        self.post_process = True

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.grayscale_value = self.grayscale_image[y, x]
            print('HSV value at ({}, {}): {}'.format(x, y, self.grayscale_value))
            self.create_mask()

    def create_mask(self):
        self.lower = max(self.grayscale_value - 40, 0)
        self.upper = min(self.grayscale_value + 40, 255)
        mask = cv2.inRange(self.grayscale_image, np.array(self.lower), np.array(self.upper))
        if self.post_process:
            cv2.erode(mask, None, iterations=2)
            cv2.dilate(mask, None, iterations=2)
        cv2.imshow('mask', mask)
        cv2.waitKey(0)

    def upper_callback(self, value):
        self.upper = value
        if self.upper < self.lower:
            self.upper = self.lower
        self.update_mask()

    def lower_callback(self, value):
        self.lower = value
        if self.lower > self.upper:
            self.lower = self.upper
        self.update_mask()

    def update_mask(self):
        mask = cv2.inRange(self.grayscale_image, np.array(self.lower), np.array(self.upper))
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
        print(self.upper)
        print(self.lower)
        cv2.createTrackbar('upper', 'control pannel', self.upper, 255, self.upper_callback)
        cv2.createTrackbar('lower', 'control pannel', self.lower, 255, self.lower_callback)
        
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()



if __name__ == '__main__':
    grayscale = grayscale()
    grayscale.main()