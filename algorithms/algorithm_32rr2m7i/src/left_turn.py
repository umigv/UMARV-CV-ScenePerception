import math
import cv2
import numpy as np
from hsv import hsv

# Left turn algorithm updated from testing at comp, !!NOT TESTED!!

class leftTurn:
    def __init__(self):
        self.last_diff_y = -3
        self.image = None
        self.hsv_image = None
        self.x = 0
        self.y = 0
        self.edge_white_x = 0
        self.edge_white_y = 0
        self.white_mask = None
        self.final = None
        self.diff_y = -3
        self.diff_x = 15
        self.yellow_mask = None
        self.yellow_found = False
        self.hsv_obj = None
        self.centroid = (None, None)
        self.width = None
        self.done = True
        self.barrel_boxes = None
        self.height = None
        self.state_1_done = False
        self.midpoint = None
        self.in_state_4 = False
        self.testing = True

    def find_slope(self, cur_x, cur_y, edge_white_x, edge_white_y):
        self.diff_y = (edge_white_y - cur_y)
        self.diff_x = (edge_white_x - cur_x)

        return self.diff_x, self.diff_y
    
    def past_stop_line(self):
        cnts, _ = cv2.findContours(self.yellow_mask[:, :self.width//2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(cnts) == 0:
            return True
        else:
            return False
            
            
    def draw_trapazoid(self):
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
        cv2.fillPoly(self.final, [pts], 0)


    def update_mask(self):
        #defining the ranges for HSV values
        self.final, dict = self.hsv_obj.get_mask(self.image) # , yolo_barrels=True, yolo_lanes=True

        self.white_mask = dict["white"]
        self.yellow_mask = dict["yellow"]
        
        # self.past_stop_line()
        self.last_diff_y = self.state_machine()
        self.find_center_of_lane()
        
        cv2.circle(self.final, self.centroid, 10, 255, -1)
        cv2.imshow("mask", self.final)
        # final_bgr = cv2.cvtColor(self.final, cv2.COLOR_GRAY2BGR)
        # combined = np.vstack((self.image, final_bgr))
        # cv2.imshow("mask", combined)
        
    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height
    
    def is_local_0s(self, mask, x, y):
        return (
        mask[y,   x]   == 0 and
        mask[y,   x+1] == 0 and
        mask[y+1, x]   == 0 and
        mask[y+1, x+1] == 0
    )
        
    
    def find_center_of_lane(self):
        if(self.centroid == (None, None)):
            return
        
        assert(self.white_mask.shape == self.final.shape)
        # start = (0, height)
        # end = (edge_white_x, edge_white_y)
        x1, y1 = self.centroid[0], self.centroid[1]
        x2, y2 = self.width//2, self.height
        
        
        #find rise and run of the
        rise = (y2 - y1)//10
        run = (x2 - x1)//10
        
        # rise is negative and run is positive 
        # rise and run are huge so divide by 10
        # print("rise, run", rise, run)
        waypoints = []
        curr_x, curr_y = x2, y2
        
        while curr_y > 30:
            # start at x2 following the slope (rise and run)
            # keeping track of the points in waypoints
            curr_x -= run # positive so subtract, update x2(bottom coordinate for next iteration) 
            curr_y -= rise #negative so add, update y2(bottom coordinate for next iteration)
            waypoints.append((curr_x,curr_y))
            # cv2.circle(self.final, (curr_x, curr_y), 5, 255, -1)
            # print(curr_x,curr_y)
            
        
        return waypoints
    
    # Initial straightaway before crossing stop line
    def state_1(self):
        # print("state_1")
        # Induce forward trajetory
        # This is a will be for the initial straightaway before we cross the stopping line
        status = self.past_stop_line()
        if (status == True):
            self.state_1_done = True
            self.state_2()
            return
        else:
            self.draw_trapazoid()
            self.centroid = (self.width//2, 40)
            # Block out the stop line with the trapazoid
            # set waypoint to directly in front of the robot
        
    

    # Already passed the yellow line. Do constant left-turn tendency until we see yellow dashed
    def state_2(self):
        # print("state_2")
        # induce a constant left turn with waypoint in top corner
        # This is for the point where we have crossed the 
        # stopping line but have yet to see the yellow
        # Also revert to this state after state 1 and if in state 2 and no yellow
        self.draw_trapazoid()
        point1 = (0, int(0.25*self.height))
        point2 = (int(0.125*self.width), self.height)
        cv2.line(self.final, (int(0.6 * self.width), 0), (self.width, self.height), 255, 10) #right line
        cv2.line(self.final, point1, point2, 255, 10) #left line
        self.centroid = (self.width // 8, 40)
        
        
     # If we see yellow dashed, align with turn lane (and check for cone)    
    def state_3(self, best_cnt):
        # print("state_3")
        # Draw lane lines to align ourselves with the turn lane
        # Anytime we see yellow dashed we should invoke this state
        
        # MAKE SURE TO CHECK FOR CONE IN FRONT
        # If cone in front and close enough go to state 4
        if self.barrel_boxes != None:
            for segment in self.barrel_boxes:
                x_min, y_min, x_max, y_max = segment
                vertices = np.array([
                    [x_min * self.width, y_min * self.height], #top-left
                    [x_max * self.width, y_min * self.height], #top right
                    [x_max * self.width, y_max * self.height], #bottom-right
                    [x_min * self.width, y_max * self.height] #bottom left
                ], dtype=np.int32)
                
                if(y_min * self.height > self.height // 2):
                    # this might be a cone that is close to us so see if its in the midele
                    self.midpoint = (x_max * self.width) - (x_min * self.width)
                    if(self.midpoint > self.width // 4 and self.midpoint < (self.width - (self.width//4))):
                        self.in_state_4 = True
                        self.centroid = self.midpoint
                        return
                else:
                    self.midpoint = None
        
        else:
            # This is the logic of state 3
            max_y = 0
            x, y = None, None
            if best_cnt is not None: 
                for point in best_cnt:
                    if point[0][1] > max_y:
                        y = point[0][1]
                        x = point[0][0]
                        max_y = y
            self.diff_x, self.diff_y = None, None
            if y is not None:
                
                while self.in_bounds(x,y) and self.white_mask[y, x] != 255:
                    y += 1
                while self.in_bounds(x,y) and self.white_mask[y, x] == 255:
                    y += 1
                edge_white_y = y
                edge_white_x = x
                
                # cv2.circle(self.final, (edge_white_x, edge_white_y), 10, 255, -1)
                
                y += 200
                y = min(self.height, y)
                while self.in_bounds(x,y) and self.white_mask[y, x] != 255:
                    x -= 1
                
                self.diff_x, self.diff_y = self.find_slope(y, x, edge_white_y, edge_white_x)
                x, y = edge_white_x, edge_white_y
                self.diff_x //= 10
                self.diff_y //= 10

                # g = math.gcd(abs(self.diff_x), abs(self.diff_y))
                # print("gcd:", g)
                # if g != 0:
                #     self.diff_x //= g
                #     self.diff_y //= g
                
                x -= self.diff_x * 5
                y -= self.diff_y * 5
                
                point_list = []
                
                # print("diff_x, diff_y:", self.diff_x, self.diff_y)
                # Exit condition in testing
                if self.testing and self.diff_x >= 0:
                    self.in_state_4 = True
                    self.centroid = (self.width//2, 40)
                    return

                self.diff_x -= 20 #Applying this so that the slope is a little more accurate

                # Use Bresenham's line algorithm to move along the slope until we hit a non-white pixel
                # Then use Bresenham's line algorithm again to move along the slope until we hit a white pixel, which should be the lane line

                # Bresenham's line algorithm implementation
                # x, y = edge_white_x, edge_white_y

                # dx = abs(self.diff_x)
                # dy = abs(self.diff_y)
                # sx = 1 if self.diff_x > 0 else -1
                # sy = 1 if self.diff_y > 0 else -1
                # err = dx - dy

                # max_steps = max(self.width, self.height)
                # steps = 0

                # # ---------- Phase 1: start in white, march until first NON-white ----------
                # while self.in_bounds(x, y) and steps < max_steps:
                #     if self.white_mask[y, x] != 255:
                #         break

                #     e2 = 2 * err
                #     if e2 > -dy:
                #         err -= dy
                #         x += sx
                #     if e2 < dx:
                #         err += dx
                #         y += sy

                #     steps += 1

                # cv2.circle(self.final, (x, y), 10, 255, -1)

                # # ---------- Phase 2: continue marching until white is hit again ----------
                # while self.in_bounds(x, y) and steps < max_steps:
                #     if self.white_mask[y, x] == 255:
                #         break

                #     e2 = 2 * err
                #     if e2 > -dy:
                #         err -= dy
                #         x += sx
                #     if e2 < dx:
                #         err += dx
                #         y += sy

                #     steps += 1

                # cv2.circle(self.final, (x, y), 10, 255, -1)


                while self.in_bounds(x,y) and x < self.width - self.diff_x and y < self.height - self.diff_y and self.is_local_0s(self.white_mask, x, y):
                    x -= self.diff_x #* 2, run
                    y -= self.diff_y #* 2, rise
                    point_list.append((x,y))
                    cv2.circle(self.final, (x, y), 5, 255, -1)
                    # reduce the diff_x and diff_y each iteration so x and y move less and less
                    # self.diff_x = max(int(self.diff_x * 0.9), min_diff_x)
                    # self.diff_y = max(int(self.diff_y * 0.9), min_diff_y)
                    
                if len(point_list) == 0:
                    self.centroid = (self.width//2, 40)
                else:
                    self.centroid = point_list[len(point_list)//2]
                self.yellow_found = True
                point1 = (0, self.height)
                point2 = (edge_white_x, edge_white_y)
                cv2.line(self.final, (x, y), (self.width, self.height), 255, 10)
                cv2.line(self.final, point1, point2, 255, 10)
        
    def state_machine(self):
        # This will decide which spot to be in
        # invariant: yellow_mask and white_mask must be set
        
        self.height, self.width = self.white_mask.shape
        if not self.state_1_done:
            # still in state 1, but once we are out of state 1 there is no way back
            self.state_1()
            return
        
        contours, _ = cv2.findContours(self.yellow_mask[:, :self.width//2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 200
        
        num_yellow_dashed = 0
        max_y = 0
        best_cnt = None

        for cnt in contours: # Looping through contours to find yellow dashed lines
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
        cap = cv2.VideoCapture('data/left_turn.mp4') #Specify an integer for webcam or other camera
        self.hsv_obj = hsv('data/left_turn.mp4') # , barrel_model_path='data/obstacles.pt', lane_model_path='data/lane_lines.pt'
        
        while cap.isOpened():
            ret, self.image = cap.read()
            if ret:
                self.height, self.width, _ = self.image.shape
                
                self.update_mask()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()


def main():
    obj = leftTurn()
    obj.run()

if __name__ == "__main__":
    main()
