import cv2
import numpy as np


class right_turn:
    def __init__(self):
        self.image = None
        self.hsv_image = None
        self.white_mask = None
        self.yellow_mask = None
        self.mask = None
        self.resized_mask = None
        self.final = None

    def update_mask(self):
        #defining the ranges for HSV values
        yellow_lower_bound = np.array([12, 95, 0])
        yellow_upper_bound = np.array([32, 255, 255])
        white_lower_bound = np.array([0, 0, 180])
        white_upper_bound = np.array([179, 103, 255])
        
        
        self.white_mask = cv2.inRange(self.hsv_image, white_lower_bound, white_upper_bound)
        self.yellow_mask = cv2.inRange(self.hsv_image, yellow_lower_bound, yellow_upper_bound) # Return a mask of HSV values within the range we specified
        
        self.white_mask = cv2.resize(self.white_mask, (self.image.shape[1], self.image.shape[0]))
        self.yellow_mask = cv2.resize(self.yellow_mask, (self.image.shape[1], self.image.shape[0]))

        #some post processing stuff to help it look smoother
        self.white_mask = cv2.erode(self.white_mask, None, iterations=2)
        self.white_mask = cv2.dilate(self.white_mask, None, iterations=2)

        self.yellow_mask = cv2.erode(self.yellow_mask, None, iterations=2)
        self.yellow_mask = cv2.dilate(self.yellow_mask, None, iterations=2)
        
        self.mask = cv2.bitwise_or(self.yellow_mask, self.white_mask)
        
        self.find_left_most_lane()
        
        # result = cv2.bitwise_and(image, image, mask=mask) #applies the mask to the base image so only masked parts of the image are shown
        self.resized_mask = cv2.resize(self.mask, (640, 480))
        # cv2.imshow("result", result)
        cv2.imshow("mask", self.resized_mask)
        
        
    def find_left_most_lane(self):
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 200 # Adjust based on noise size
        hsv_lanes = np.zeros_like(self.mask) #new occupancy grid that is blank
        
        min_x = float("inf")
        best_cnt = None
        max_y = 0
        
        for cnt in contours: # looping through contours
            if cv2.contourArea(cnt) > min_area:
                #find left most lane
                print(cnt[0, 0, 0]) # x-values of each cnt
                print(cnt[0, 0, 1]) # y-values of each cnt
                
                if cnt[0, 0, 1] > max_y:
                    max_y = cnt[0, 0, 1]
                    min_x = cnt[0, 0, 0]
                    best_cnt = (min_x, max_y)
                
                cv2.drawContours(hsv_lanes, [cnt], -1, 255, thickness=cv2.FILLED)
                cv2.circle(self.final, best_cnt, 50, 255, -1)
                
                cv2.line(self.final, (0, best_cnt[1]), (0, self.mask.shape[0]), 255, 10)
                cv2.line(self.final, (best_cnt[0], best_cnt[1]), (0, best_cnt[1]), 255, 10)
        
    # Make main function that gets the image path and passes it into update mask as an HSV converted image
    # use cv2.imread(filename) then put the output into cv2.cvtcolor(image, cv2.RBGTOHSV)

    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, self.image = cap.read()
            if ret:
                height, width, _ = self.image.shape
                right_half_vid = self.image[:, width // 2:]
                self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            
                self.update_mask()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break


def main():
    obj = right_turn()
    obj.run()


if __name__ == "__main__":
    main()
