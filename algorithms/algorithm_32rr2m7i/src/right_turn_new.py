import cv2
import numpy as np
from hsv import hsv


class RightTurn:
    def __init__(self, debug = False):
        self.image = None
        self.hsv_image = None
        self.white_mask = None
        self.yellow_mask = None
        self.final = None
        
        self.yellow_found = False

        self.hsv_obj = None

        self.centroid = (None, None)

        self.width = None
        self.height = None

        self.state_1_done = False
        self.state_4_done = False

        self.debug = debug

    def draw_trapezoid(self):
        top_width_start = self.width // 4  # Narrower top
        top_width_end = self.width - (self.width // 4)
        bottom_width_start = 0  # Wider base
        bottom_width_end = self.width - (self.width // 9)

        # Define the trapezoid points
        pts = np.array([
            [top_width_start, 400],  # Top-left
            [top_width_end, 400],    # Top-right
            [bottom_width_end, self.height],      # Bottom-right
            [bottom_width_start, self.height]     # Bottom-left
        ], dtype=np.int32)

        # Fill the trapezoid with 0 in the mask
        if self.debug:
            print("Trapezoid drawn")
        cv2.fillPoly(self.final, [pts], 0)

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
        self.final, dict = self.hsv_obj.get_mask(self.image)
        
        self.white_mask = dict["white"]
        self.yellow_mask = dict["yellow"]
        
        # self.find_center_of_lane()
        cv2.imshow("mask_final", self.final)
        final_bgr = cv2.cvtColor(self.final, cv2.COLOR_GRAY2BGR)
        combined = np.vstack((self.image, final_bgr))
        cv2.imshow("mask_combined", combined)

    def find_center_of_lane(self):
        pass

    def state_1(self):
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

    def state_2(self):
        # state2: in the case we can't see yellow dashed line but past stop line
        # start right movement
        if self.debug:
            print("state 2")

        # induce a constant left turn with waypoint in top corner
        # This is for the point where we have crossed the 
        # stopping line but have yet to see the yellow
        # Also revert to this state after state 1 and if in state 2 and no yellow
        # point1 = (0, int(0.25*self.height))
        # point2 = (int(0.125*self.width), self.height)
        # cv2.line(self.final, (int(0.6 * self.width), 0), (self.width, self.height), 255, 10) #right line
        # cv2.line(self.final, point1, point2, 255, 10) #left line

        self.draw_trapezoid()

        top_middle = (int(0.4 * self.width), 0)
        bottom_left = (0, self.height)
        cv2.line(self.final, top_middle, bottom_left, 255, 10)

        right_middle = (self.width, int(0.25 * self.height))
        bottom_middle = (int(0.875*self.width), self.height)
        cv2.line(self.final, right_middle, bottom_middle, 255, 10)

        self.centroid = ((self.width // 8) * 7, 40)

    def state_3(self, best_cnt):
        # state3: the case where we're mid-turn and can see the yellow dashed line
        # using cv2 contours, detect and draw temp lane lines
        if self.debug:
            print("state 3")

        max_y = 0
        x, y = None, None

        # find x, y of lowest point of contour
        if best_cnt is not None:
            for point in best_cnt:
                if point[0][1] > max_y:
                    y = point[0][1]
                    x = point[0][0]
                    max_y = y

        if (x is not None) and (y is not None):
            x2 = max(0, x - 150)
            y2 = y

            while y2 > 0 and self.final[y2, x2] != 255:
                y2 -= 1 # bring up to bottom of white line
            while y2 > 0 and self.final[y2, x2] != 0:
                y2 -= 1 # bring up to top of white line

            if self.debug:
                cv2.circle(self.final, (x, y), 5, 128, -1)
                cv2.circle(self.final, (x2, y2), 5, 128, -1)
            
            min_x_dist = 40
            # invalid_points = ((y2 == 0) or y > self.height // 3) or (x - x2 < min_x_dist)
            invalid_points = (y2 == 0 and (y > self.height // 8)) or (x - x2 < min_x_dist)

            if invalid_points: # white line is probably gone, so set centroid up ahead
                self.centroid  = (self.width // 2, 40)
            else: # slope logic
                diff_x = x - x2
                diff_x //= 10
                diff_y = y - y2
                diff_y //= 10

                point_list = []
                
                initial_jump_factor = 2
                curr_x, curr_y = x + (diff_x * initial_jump_factor), y + (diff_y * initial_jump_factor)

                while (curr_x > 0 and curr_x < self.width - diff_x) and (curr_y > 0 and curr_y < self.height - diff_y) and (self.final[curr_y, curr_x] == 0):
                    curr_x += diff_x
                    curr_y += diff_y
                    point_list.append((curr_x, curr_y))

                [cv2.circle(self.final, point, 5, 128, -1) for point in point_list if self.debug]

                if (len(point_list) // 2) >= 0 and len(point_list) // 2 < len(point_list):
                    self.centroid = point_list[len(point_list) // 2]

                    # print(f"{self.centroid[1] / self.height}")
                    # print(f"width: {self.width}, height {self.height}")
                    # print(self.centroid)
                    # if self.centroid[1] > ((self.height // 5) * 4):
                    #     print("centroid is too low, sending it back")
                    #     self.centroid = (self.width // 2, 40)

    def state_machine(self):
        if not self.state_1_done:
            # still in state 1, but once we are out of state 1 there is no way back
            self.state_1()
            return
        
        contours, _ = cv2.findContours(self.yellow_mask[:, :self.width//2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 200
        best_cnt = None
        max_y = 0
        
        num_yellow_dashed = 0
        for cnt in contours: # Looping through contours
            if cv2.contourArea(cnt) > min_area:
                num_yellow_dashed += 1
                
                if cnt[0, 0, 1] > max_y and cnt[0, 0, 0] < self.width // 2 and cnt[0, 0, 1] < self.height // 2:
                    max_y = cnt[0, 0, 1]
                    best_cnt = cnt
                    
        if num_yellow_dashed == 0 or (best_cnt is None):
            self.state_2()
            return
        else:
            self.state_3(best_cnt)

    def run(self):
        cap = cv2.VideoCapture('data/right_turn1.mp4')
        self.hsv_obj = hsv('data/trimmed.mov')
        
        while cap.isOpened():
            ret, self.image = cap.read()
            if ret:
                self.height, self.width, _ = self.image.shape
                
                self.update_mask()
                self.state_machine()

                cv2.circle(self.final, self.centroid, 5, 255, -1)

                cv2.namedWindow("Final", cv2.WINDOW_NORMAL)
                cv2.imshow("Final", self.final)
                cv2.namedWindow("Yellow", cv2.WINDOW_NORMAL)
                cv2.imshow("Yellow", self.yellow_mask)
                cv2.namedWindow("White", cv2.WINDOW_NORMAL)
                cv2.imshow("White", self.white_mask)

                if self.debug:
                    print()
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
        
    def run_frame(self, hsv_indentifier, frame):
        if self.hsv_obj is None:
            self.hsv_obj = hsv(hsv_indentifier)
        
        self.image = frame
        self.height, self.width, _ = self.image.shape
    
        self.update_mask()

def main():
    obj = RightTurn(debug = False)
    obj.run()

if __name__ == "__main__":
    main()