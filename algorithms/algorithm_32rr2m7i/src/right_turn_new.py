import cv2
import numpy as np
from hsv import hsv


class RightTurn:
    def __init__(self):
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

        self.midpoint = None

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
        print("Trapezoid drawn")
        cv2.fillPoly(self.final, [pts], 0)

    def past_stop_line(self):
        cnts, _ = cv2.findContours(self.yellow_mask[:, :self.width//2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        
        # self.past_stop_line()
        
        # if(self.done == False):
        # self.last_diff_y = self.find_left_most_lane()
        # else:
        #     self.draw_trapazoid()
        #     self.centroid = (self.width//2, 40)
        self.find_center_of_lane()
        cv2.imshow("mask", self.final)
        final_bgr = cv2.cvtColor(self.final, cv2.COLOR_GRAY2BGR)
        combined = np.vstack((self.image, final_bgr))
        cv2.imshow("mask", combined)

    def find_center_of_lane(self):
        pass

    def state_1(self):
        # Induce forward trajectory
        # This is a will be for the initial straightaway before we cross the stopping line
        print("state 1")

        status = self.past_stop_line()
        self.draw_trapezoid()
        if (status == True):
            print("past_stop_line was true")
            self.state_1_done = True
            self.state_2()
            return
        else:
            print("past_stop_line was false")
            self.centroid = (self.width//2, 40)
            # Block out the stop line with the trapazoid
            # set waypoint to directly in front of the robot

    def state_2(self):
        # state2: in the case we can't see yellow dashed line but past stop line
        # start right movement
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
        print("state 3")

    def state_machine(self):
        self.height, self.width = self.white_mask.shape
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
                    
        if num_yellow_dashed == 0:
            self.state_2()
            return
        else:
            self.state_3(best_cnt)

    def run(self):
        cap = cv2.VideoCapture('data/right_turn1.mp4')
        self.hsv_obj = hsv('data/trimmed.mov')
        
        # self.hsv_obj = self.hsv_obj.tune('data/trimmed.mov')
        
        while cap.isOpened():
            ret, self.image = cap.read()
            if ret:
                self.height, self.width, _ = self.image.shape
                
                self.update_mask()
                self.state_machine()

                cv2.circle(self.final, self.centroid, 5, 255, -1)

                cv2.imshow("Final", self.final)
                cv2.imshow("Yellow", self.yellow_mask)
                cv2.imshow("White", self.yellow_mask)

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
    obj = RightTurn()
    obj.run()

if __name__ == "__main__":
    main()