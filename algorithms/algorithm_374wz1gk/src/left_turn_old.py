import cv2
import numpy as np
from hsv import hsv

# original left turn algo, found it was a little too simple for what we saw at comp

class left_turn:
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
        self.height = None
        

    def find_slope(self, cur_x, cur_y, edge_white_x, edge_white_y):
        self.diff_y = (edge_white_y - cur_y)
        self.diff_x = (edge_white_x - cur_x)

        return self.diff_x, self.diff_y
    
    def past_stop_line(self):
        cnts, _ = cv2.findContours(self.yellow_mask[:, :self.width//2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(cnts) == 0:
            self.done = False
            
            
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
    
    
    
    def find_left_most_lane(self):
        print("WHITE MASK SHAPE", self.white_mask.shape)
        print("FINAL SHAPE", self.final.shape)
        assert(self.white_mask.shape == self.final.shape)
        contours, _ = cv2.findContours(self.yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 200 # Adjust based on noise size
        # hsv_lanes = np.zeros_like(self.mask) # New occupancy grid that is blank
        self.height, self.width = self.white_mask.shape
        
        if len(contours) != 0:
            self.done = False
        
        
        if self.done:
            self.draw_trapazoid()
            self.centroid = (self.width//2, 40)
            return -3
        
        
        best_cnt = None
        max_y = 0
        
        for cnt in contours: # Looping through contours
            if cv2.contourArea(cnt) > min_area:
                # Find left most lane
                if cnt[0, 0, 1] > max_y and cnt[0, 0, 0] < self.width // 2 and cnt[0, 0, 1] < self.height // 2:
                    max_y = cnt[0, 0, 1]
                    best_cnt = cnt
                
                # cv2.drawContours(hsv_lanes, [cnt], -1, 255, thickness=cv2.FILLED)
        
        max_y = 0
        x, y = None, None
        if best_cnt is not None:
            for point in best_cnt:
                if point[0][1] > max_y:
                    y = point[0][1]
                    x = point[0][0]
                    max_y = y
                
        self.diff_x, self.diff_y = None, None

        if self.yellow_found is False:
            #right line
            self.draw_trapazoid()
            point1 = (0, int(0.5*self.height))
            point2 = (int(1), self.height)
            cv2.line(self.final, (int(0.8 * self.width), 0), (self.width, self.height), 255, 10)
            #left line
            cv2.line(self.final, point1, point2, 255, 10)
            self.centroid = (self.width // 4, 40)
            
            
            # self.find_center_of_lane()
        if y is not None:
            # print(x, y)
            
            # cv2.circle(final, (x, y), 100, 255, -1)

            edge_white_x = x
            edge_white_y = y
            
            # cv2.circle(self.final, (x, y), 15, 255, -1)
            
            x -= 150
            x = max(0, x)
            while y < self.height and self.white_mask[y, x] != 255:
                y += 1
                # cv2.circle(self.final, (x, y), 3, 255, -1)
            
            self.diff_x, self.diff_y = self.find_slope(x, y, edge_white_x, edge_white_y)    
            x, y = edge_white_x, edge_white_y
            self.diff_x //= 10
            self.diff_y //= 10
            # print(f"diffx {self.diff_x}, diffy {self.diff_y}")
            
            x += self.diff_x * 5
            y += self.diff_y * 5
            
            point_list = []
            
            while x > 0 and y > 0 and x < self.width - self.diff_x and y < self.height - self.diff_y and self.white_mask[y, x] == 0:
                # cv2.circle(self.final, (x, y), 5, 255, -1)

                x += self.diff_x #* 2, run
                y += self.diff_y #* 2, rise
                
                point_list.append((x,y))
                
            
            if(len(point_list) > 2):
                self.centroid = point_list[len(point_list)//2]
                
            
            # cv2.circle(self.final, self.centroid, 25, 255, -1)
                
            # self.find_center_of_lane()
            if(self.diff_y > -20 and self.diff_y < 0 and abs(self.last_diff_y - self.diff_y) < 10 and (self.white_mask[y, x] != 0)):
                self.yellow_found = True
                point1 = (0, self.height)
                point2 = (edge_white_x, edge_white_y)
                cv2.line(self.final, (x, y), (self.width, self.height), 255, 10)
                cv2.line(self.final, point1, point2, 255, 10)
                
            
            if abs(self.last_diff_y - self.diff_y) >= 10:
                self.done = True
                
            
            return self.diff_y
        
        else:
            return -3


    def run(self):
        cap = cv2.VideoCapture('data/left_turn_full.mp4')
        self.hsv_obj = hsv('data/trimmed.mov')
        
        self.hsv_obj = tune('data/trimmed.mov')
        
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
    obj = left_turn()
    obj.run()

if __name__ == "__main__":
    main()