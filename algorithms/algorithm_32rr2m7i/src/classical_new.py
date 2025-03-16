import cv2
import numpy as np

width = 640
height = 480
class hsv:
    def __init__(self):
        self.image_path = 'yellow_dashed_center_2.png'
        self.image = cv2.imread(self.image_path)
        self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.hsv_value = None
        self.h_upper = None
        self.h_lower = None
        self.s_upper = None
        self.s_lower = None
        self.v_upper = None
        self.v_lower = None
        self.post_process = True

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.hsv_value = self.hsv_image[y, x]
            print('HSV value at ({}, {}): {}'.format(x, y, self.hsv_value))
            self.create_mask()
            
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
        resized_mask = cv2.resize(mask, (width, height))
        cv2.imshow('mask', resized_mask)
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
        resized_mask = cv2.resize(mask, (width, height))
        cv2.imshow('mask', resized_mask)
        

        resized_result = cv2.resize(result, (width, height))
        cv2.imshow('result', resized_result)

        # Find the closest row from the bottom where the lane lines start
        closest_row = self.find_closest_row(mask)
        if closest_row is not None:
            l1, l2 = self.find_line_cols(closest_row, mask)
            print(f"Closest row from the bottom where lane lines start: {closest_row}")
            cv2.line(result, (0, closest_row), (result.shape[1], closest_row), (0, 255, 0), 2)
            cv2.line(result, (l1, closest_row), (0, result.shape[0]), (255, 0, 0), 2)  # Line to bottom-left corner
            cv2.line(result, (l2, closest_row), (result.shape[1], result.shape[0]), (255, 0, 0), 2)  # Line to bottom-right corner
            cv2.line(mask, (l1, closest_row), (0, result.shape[0]), (255, 0, 0), 10)  # Line to bottom-left corner
            cv2.line(mask, (l2, closest_row), (result.shape[1], result.shape[0]), (255, 0, 0), 10)  # Line to bottom-right corner
           
            
        

            resized_result = cv2.resize(result, (width, height))
            cv2.imshow('result_with_line', resized_result)
            resized_mask = cv2.resize(mask, (width, height))
            cv2.imshow('mask', resized_mask)

            
        
    def main(self):
        width = 640
        height = 480
        resized_image = cv2.resize(self.image, (width, height))

        cv2.imshow('raw image', resized_image)
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