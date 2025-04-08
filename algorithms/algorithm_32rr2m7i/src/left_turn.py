import cv2
import numpy as np
from hsv import hsv


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
        

    def find_slope(self, cur_x, cur_y, edge_white_x, edge_white_y):
        self.diff_y = (edge_white_y - cur_y)
        self.diff_x = (edge_white_x - cur_x)

        return self.diff_x, self.diff_y

    def update_mask(self):
        #defining the ranges for HSV values
        self.final, dict = self.hsv_obj.get_mask(self.image)
        
        self.white_mask = dict["white"]
        self.yellow_mask = dict["yellow"]
        # result = cv2.bitwise_and(image, image, mask=mask) #applies the mask to the base image so only masked parts of the image are shown
        # resized_mask = cv2.resize(final_mask, (640, 480))
        # cv2.imshow("result", result)
        
        self.last_diff_y = self.find_left_most_lane()
        cv2.imshow("mask", self.final)
        
    
    def find_center_of_lane(self, start, end):
        assert(self.white_mask.shape == self.final.shape)
        # start = (0, height)
        # end = (edge_white_x, edge_white_y)
        x1, y1 = start[0], start[1]
        x2, y2 = end[0], end[1]
        
        rise = y2 - y1
        run = x2 - x1
        waypoints = []
        curr_x, curr_y = x1, y1
        while curr_x < x2 and curr_y < y2:
            # find normal to line at curr_x, curr_y
            # keep going until you intercept the other lane line (use white mask, stop when you hit white)
            # find centroid, and append to waypoints
            normal = (-run, rise)
            temp_x, temp_y = curr_x, curr_y
            while(temp_x < x2 and temp_y < y2 and self.white_mask[temp_y, temp_x] == 0):
                temp_x += normal[0]
                temp_y += normal[1]
            centroid = ((curr_x+temp_x)//2, (curr_y+temp_y)//2)
            cv2.circle(self.final, centroid, 10, 255, -1)
            waypoints.append(centroid)
            curr_x += 5*run
            curr_y += 5*rise
            print(waypoints)
        
        return waypoints
    
    
    
    def find_left_most_lane(self):
        assert(self.white_mask.shape == self.final.shape)
        contours, _ = cv2.findContours(self.yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 200 # Adjust based on noise size
        # hsv_lanes = np.zeros_like(self.mask) # New occupancy grid that is blank
        height, width = self.white_mask.shape
        
        best_cnt = None
        max_y = 0
        
        for cnt in contours: # Looping through contours
            if cv2.contourArea(cnt) > min_area:
                # Find left most lane
                if cnt[0, 0, 1] > max_y and cnt[0, 0, 0] < width // 2 and cnt[0, 0, 1] < height // 2:
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
            point1 = (0, int(0.5*height))
            point2 = (int(0.125*width), height)
            cv2.line(self.final, (int(0.8 * width), 0), (width, height), 255, 10)
            #left line
            cv2.line(self.final, point1, point2, 255, 10)
            
            self.find_center_of_lane(point1, point2)
        if y is not None:
            # print(x, y)
            
            # cv2.circle(final, (x, y), 100, 255, -1)

            edge_white_x = x
            edge_white_y = y
            
            # cv2.circle(self.final, (x, y), 15, 255, -1)
            
            x -= 150
            x = max(0, x)
            
            
            
            while y < height and self.white_mask[y, x] != 255:
                y += 1
                # cv2.circle(self.final, (x, y), 3, 255, -1)
            
            self.diff_x, self.diff_y = self.find_slope(x, y, edge_white_x, edge_white_y)    

                
            x, y = edge_white_x, edge_white_y
            self.diff_x //= 10
            self.diff_y //= 10
            
            
            
            
            print(f"diffx {self.diff_x}, diffy {self.diff_y}")
            
            x += self.diff_x * 2
            y += self.diff_y * 2
            
            
            

            
            while x > 0 and y > 0 and x < width - self.diff_x and y < height - self.diff_y and self.white_mask[y, x] == 0:
                # cv2.circle(self.final, (x, y), 5, 255, -1)

                x += self.diff_x #* 2
                y += self.diff_y #* 2
                
            if(self.diff_y > -40 and self.diff_y < 0 and abs(self.last_diff_y - self.diff_y) < 10 and (self.white_mask[y, x] != 0)):
                self.yellow_found = True
                point1 = (0, height)
                point2 = (edge_white_x, edge_white_y)
                cv2.line(self.final, (x, y), (width, height), 255, 10)
                cv2.line(self.final, point1, point2, 255, 10)
                self.find_center_of_lane(point1, point2)
            return self.diff_y
        
        # if self.yellow_found is False:
        #     #right line
        #     cv2.line(self.final, (int(0.8 * width), 0), (width, height), 255, 10)
        #     #left line
        #     cv2.line(self.final, (0, int(0.5*height)), (int(0.125*width), height), 255, 10)
        
        else:
            return -3
        
        
                
            
            
        
    # Make main function that gets the image path and passes it into update mask as an HSV converted image
    # use cv2.imread(filename
    # ) then put the output into cv2.cvtcolor(image, cv2.RBGTOHSV)


    def adjust_gamma(self, image, gamma=0.4):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        image = cv2.LUT(image, table)
        return image


    def run(self):
        cap = cv2.VideoCapture('data/trimmed.mov')
        self.hsv_obj = hsv('data/trimmed.mov')
        
        while cap.isOpened():
            ret, self.image = cap.read()
            if ret:
                self.image = self.adjust_gamma(self.image)
                self.height, self.width, _ = self.image.shape
                
                self.update_mask()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()


def main():
    obj = left_turn()
    obj.run()

if __name__ == "__main__":
    main()
