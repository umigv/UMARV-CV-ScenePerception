import cv2
import numpy as np
from hsv import hsv

class ParkingPullOut:
    def __init__(self, debug = False):
        self.image = None
        self.hsv_image = None

        self.white_mask = None
        self.yellow_mask = None

        self.final = None

        self.hsv_obj = None

        self.centroid = (None, None)

        self.width = None
        self.height = None

        self.state_1_done = False
        self.state_2_done = False
        self.state_3_done = False

        self.min_area = 200

        self.look_for_barrels = True

        self.debug = debug
        
        # This variable will change depending on what side the barrel is on. The default will be left
        self.left = True
    
    def barrel_left(self):
        ret, img = self.cap.read()
        height, width = img.shape[:2]
        img = img[:, int(width/2) : width]
        height, width = img.shape[:2]
        img = img[:int(height * 1/4), :]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([23,96,231])
        upper_yellow = np.array([179,255,255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        middle = int(width/2)
        left = yellow_mask[:, :middle]
        right = yellow_mask[:, middle:width]
        left_count = cv2.countNonZero(left)
        right_count = cv2.countNonZero(right)
        
        result = cv2.bitwise_and(img, img, mask=yellow_mask)
        
        
        # Wait 1ms and check if 'q' is pressed
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break  
        # blank_image = np.zeros((width, height, 3))
        

        # blank = cv2.bitwise_or(img, img, left)
        # blank_2 = cv2.bitwise_or(img_2, img_2, right)
        cv2.imshow("result", result)
        # cv2.imshow(img, "left")
        # cv2.imshow(img_2, "right")

        if(left_count > right_count):
            print("Right to left= True")
            return True
        print("Right to left= False")
        return False

    def draw_trapezoid(self):
        top_width_start = self.width // 2.2  # Narrower top
        top_width_end = self.width - (self.width // 2.2)
        bottom_width_start = self.width // 3  # Wider base
        bottom_width_end = self.width - (self.width // 3)

        # Define the trapezoid points
        pts = np.array([
            [top_width_start, 400],               # Top-left
            [top_width_end, 400],                 # Top-right
            [bottom_width_end, self.height],      # Bottom-right
            [bottom_width_start, self.height]     # Bottom-left
        ], dtype=np.int32)

        # Fill the trapezoid with 0 in the mask
        if self.debug:
            print("Trapezoid drawn")
            
        cv2.fillPoly(self.final, [pts], 0)

    # Use this to check if past the dotted yellow lines
    def past_stop_line(self):
        cnts, _ = cv2.findContours(self.yellow_mask[:, :self.width//2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if self.debug:
            print(f"Yellow contours (count): {len(cnts)}")

        if len(cnts) == 0:
            return True
        else:
            return False
        

    def update_mask(self):
        #defining the ranges for HSV values
        self.final, dict = self.hsv_obj.get_mask(self.image, yolo_barrels=self.look_for_barrels and (not self.debug))

        # print(dict)
        
        self.white_mask = dict["white"]
        self.yellow_mask = dict["yellow"]

        # final_bgr = cv2.cvtColor(self.final, cv2.COLOR_GRAY2BGR)
        # combined = np.hstack((self.image, final_bgr))
        # cv2.namedWindow("Combined Image", cv2.WINDOW_NORMAL)
        # cv2.imshow("Combined Image", self.final)        
    
    
    
    
    
    
    
    # state 1 : before the turn, while yellow dotted line is still visible. Also detect which way the barrel is
    def state_1(self):
        # Check on which side the barrel is on
        self.left = self.barrel_left()
        
        # Set the way point to be directly ahead
        # Induce forward trajectory
        # This is a will be for the initial straightaway before we cross the stopping line

        if self.debug:
            print("state 1")

        
        status = self.past_stop_line()
        
        self.draw_trapezoid()
        if (status == True):
            self.state_1_done = True
            self.state_2()
            return
        else:
            self.centroid = (self.width // 2, 40)
            # Block out the stop line with the trapazoid
            # set waypoint to directly in front of the robot
        
        
    


    # state 2: get the robot to turn as soon as we pass the dotted yellow line for the first time!
    # Get the robot to turn the direction 
    def state_2(self):
        
        if self.debug:
            print("state_2")
        
        
        # Case if barrel is to the right
        # Constant right turn taken from right turn
        self.look_for_barrels = True

        if self.left == False:
            self.draw_trapezoid()

            top_middle = (int(0.4 * self.width), 0)
            bottom_left = (0, self.height)
            cv2.line(self.final, top_middle, bottom_left, 255, 10)

            right_middle = (self.width, int(0.25 * self.height))
            bottom_middle = (int(0.875*self.width), self.height)
            cv2.line(self.final, right_middle, bottom_middle, 255, 10)

            self.centroid = ((self.width // 5) * 3, int((self.height // 8) * 2.5))
        # Case if barrel is to the left
        # Constant left turn is taken from left turn
        else:
            self.draw_trapazoid()
            point1 = (0, int(0.25*self.height))
            point2 = (int(0.125*self.width), self.height)
            cv2.line(self.final, (int(0.6 * self.width), 0), (self.width, self.height), 255, 10) #right line
            cv2.line(self.final, point1, point2, 255, 10) #left line
            self.centroid = (self.width // 8, 40)

        # Condition to stop once barrel can be seen and transition to state 3
        # Put logic here
        if self.hsv_obj.barrel_boxes is not None:
            self.state_2_done = True



            
        
        


    # state 3: after turn, stop in front of the barrel.
    def state_3(self, yellow_cnts):
        # look for barrel
        if self.hsv_obj.barrel_boxes is not None:
            for segment in self.hsv_obj.barrel_boxes:
                x_min, y_min, x_max, y_max = segment
                vertices = np.array([
                    [x_min * self.width, y_min * self.height], #top left
                    [x_max * self.width, y_min * self.height], #top right
                    [x_max * self.width, y_max * self.height], #bottom right
                    [x_min * self.width, y_max * self.height] #bottom left
                ], dtype=np.int32)
                    
                if(y_min * self.height > self.height // 2):
                    # this might be a cone that is close to us so see if its in the middle
                    midpoint = (x_max * self.width) - (x_min * self.width)
                    if(midpoint > self.width // 4 and midpoint < (self.width - (self.width//4))):
                        self.centroid = midpoint
                        return

        # normal state 3
        min_y = self.height - 1
        top_yellow_point = None

        # look for top yellow point
        for cnt in yellow_cnts:
            if cv2.contourArea(cnt) > self.min_area:
                for point in cnt:
                   if point[0, 1] < min_y:
                        top_yellow_point = (point[0, 0], point[0, 1])
                        min_y = point[0, 1]
        if top_yellow_point is None:
            self.centroid  = (self.width // 2, 40)
            return
            
        min_y = self.height - 1
        top_white_point = None
        white_cnts, _ = cv2.findContours(self.white_mask[:self.height//2, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

         # look for top white point
        for cnt in white_cnts:
            if cv2.contourArea(cnt) > self.min_area:
                for point in cnt:
                    if point[0, 1] < min_y:
                        top_white_point = (point[0, 0], point[0, 1])
                        min_y = point[0, 1]
        if top_white_point is None:
            self.centroid  = (self.width // 2, 40)
            return

        avg_x = (top_yellow_point[0] + top_white_point[0]) // 2
        avg_y = (top_yellow_point[1] + top_white_point[1]) // 2
        if self.debug:
            cv2.line(self.final, top_white_point, top_yellow_point, 128, 10)

        self.centroid = (avg_x, avg_y)

    def run(self):
            cap = cv2.VideoCapture("data/right_turn_cropped.mp4")
            self.hsv_obj = hsv("data/right_turn_cropped.mp4")

            # "white": {
            #     "h_upper": 179,
            #     "h_lower": 0,
            #     "s_upper": 218,
            #     "s_lower": 0,
            #     "v_upper": 255,
            #     "v_lower": 212
            # },
            # "yellow": {
            #     "h_upper": 179,
            #     "h_lower": 23,
            #     "s_upper": 255,
            #     "s_lower": 150,
            #     "v_upper": 255,
            #     "v_lower": 200
            # }
            # backup of values from json

            # self.hsv_obj.tune("white")
            # self.hsv_obj.tune("yellow")
            
            while cap.isOpened():
                ret, self.image = cap.read()
                if ret:
                    self.height, self.width, _ = self.image.shape
                    
                    self.update_mask()
                    self.state_machine()

                    cv2.circle(self.final, self.centroid, 5, 255, -1)

                    cv2.namedWindow("Final Mask", cv2.WINDOW_NORMAL)
                    cv2.imshow("Final Mask", self.final)
                    cv2.namedWindow("Yellow Mask", cv2.WINDOW_NORMAL)
                    cv2.imshow("Yellow Mask", self.yellow_mask)
                    cv2.namedWindow("White Mask", cv2.WINDOW_NORMAL)
                    cv2.imshow("White Mask", self.white_mask)

                    if self.debug:
                        print()
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            cap.release()
            cv2.destroyAllWindows()


    # State machine to get things running
    def state_machine(self):
        if not self.state_1_done:
            # still in state 1, but once we are out of state 1 there is no way back
            self.state_1()
            return
            
        contours, _ = cv2.findContours(self.yellow_mask[:, :self.width//2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = None
        max_y = 0
            
        num_yellow_dashed = 0
        for cnt in contours: # Find lowermost yellow contour
            if cv2.contourArea(cnt) > self.min_area:
                num_yellow_dashed += 1
                    
                if cnt[0, 0, 1] > max_y: # and cnt[0, 0, 1] < self.height // 2:
                    max_y = cnt[0, 0, 1]
                    best_cnt = cnt
                        
        if (num_yellow_dashed == 0 or (best_cnt is None)) and not self.state_2_done: # state 2
            self.state_2()
            return
        elif not self.state_3_done: # to start state 3
            self.state_2_done = True
            self.look_for_barrels = True
            self.state_3(contours)


def main():
    obj = ParkingPullOut(debug = False)
    obj.run()

if __name__ == "__main__":
    main()