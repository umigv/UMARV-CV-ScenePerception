import cv2
import numpy as np
from drivable_area.hsv import hsv


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
        self.final, dict = self.hsv_obj.get_mask(self.image)
        
        self.white_mask = dict["white"]
        self.yellow_mask = dict["yellow"]
        
        self.past_stop_line()
        
        # if(self.done == False):
        self.last_diff_y = self.find_left_most_lane()
        # else:
        #     self.draw_trapazoid()
        #     self.centroid = (self.width//2, 40)
        self.find_center_of_lane()
        cv2.imshow("mask", self.final)
        final_bgr = cv2.cvtColor(self.final, cv2.COLOR_GRAY2BGR)
        combined = np.vstack((self.image, final_bgr))
        cv2.imshow("mask", combined)
        
    
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
    
    
    def state_1(self):
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
        
    
    def state_2(self):
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
        
        
        
    def state_3(self, best_cnt):
        # Draw lane lines to align outselved with the turn lane
        # Anytime we see yellow dashed we should invoke this state
        
        # MAKE SURE TO CHECK FOR CONE IN FRONT
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
            edge_white_x = x
            edge_white_y = y
            
            x -= 150
            x = max(0, x)
            while y < self.height and self.white_mask[y, x] != 255:
                y += 1
            
            self.diff_x, self.diff_y = self.find_slope(x, y, edge_white_x, edge_white_y)    
            x, y = edge_white_x, edge_white_y
            self.diff_x //= 10
            self.diff_y //= 10
            
            x += self.diff_x * 5
            y += self.diff_y * 5
            
            point_list = []
            
            while x > 0 and y > 0 and x < self.width - self.diff_x and y < self.height - self.diff_y and self.white_mask[y, x] == 0:
                x += self.diff_x #* 2, run
                y += self.diff_y #* 2, rise
                point_list.append((x,y))
            self.centroid = point_list[len(point_list)//2]
            self.yellow_found = True
            point1 = (0, self.height)
            point2 = (edge_white_x, edge_white_y)
            cv2.line(self.final, (x, y), (self.width, self.height), 255, 10)
            cv2.line(self.final, point1, point2, 255, 10)
        
        
    
    
        
    def state_machine(self):
        # This will decide which spot to be in
        # invariant: yellow_mask and white_mask must be set
        assert(self.white_mask != None)
        assert(self.yellow_mask != None)
        
        self.height, self.width = self.white_mask.shape
        if not self.state_1_done:
            # still in state 1, but once we are out of state 1 there is no way back
            self.state_1()
            return
        
        contours, _ = cv2.findContours(self.yellow_mask[:, :self.width//2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 200
        
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
        cap = cv2.VideoCapture('data/left_turn_full.mp4')
        self.hsv_obj = hsv('data/trimmed.mov')
        
        self.hsv_obj = self.hsv_obj.tune('data/trimmed.mov')
        
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
        
    def run_frame(self, hsv_indentifier, frame):
        if self.hsv_obj is None:
            self.hsv_obj = hsv(hsv_indentifier)
        
        self.image = frame
        self.height, self.width, _ = self.image.shape
    
        self.update_mask()


def main():
    obj = leftTurn()
    obj.run()

if __name__ == "__main__":
    main()
