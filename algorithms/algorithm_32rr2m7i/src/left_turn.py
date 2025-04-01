import cv2
import numpy as np


class left_turn:
    def __init__(self):
        self.last_diff_y = -3
        self.image = None
        self.hsv_image = None
        self.x = 0
        self.y = 0
        self.edge_white_x = 0
        self.edge_white_y = 0
        self.mask = None
        self.white_mask = None
        self.final = None
        self.diff_y = -3
        self.diff_x = 15
        self.yellow_mask = None
        self.mask = None
        self.yellow_found = False

    def find_slope(self, cur_x, cur_y, edge_white_x, edge_white_y):
        self.diff_y = (edge_white_y - cur_y)
        self.diff_x = (edge_white_x - cur_x)

        return self.diff_x, self.diff_y

    def update_mask(self):
        #defining the ranges for HSV values
        yellow_lower_bound = np.array([0, 39, 227])
        yellow_upper_bound = np.array([93, 255, 255])
        white_lower_bound = np.array([0, 0, 201])
        white_upper_bound = np.array([179, 70, 255])
        
        
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
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 200 # Adjust based on noise size
        self.final = np.zeros_like(self.mask)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(self.final, [cnt], -1, 255, thickness=cv2.FILLED)
        
        # result = cv2.bitwise_and(image, image, mask=mask) #applies the mask to the base image so only masked parts of the image are shown
        # resized_mask = cv2.resize(final_mask, (640, 480))
        # cv2.imshow("result", result)
        
        self.last_diff_y = self.find_left_most_lane()     
        cv2.imshow("mask", self.final)
        
    

    
    
    def find_left_most_lane(self):
        assert(self.mask.shape == self.white_mask.shape == self.final.shape)
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
            cv2.line(self.final, (int(0.8 * width), 0), (width, height), 255, 10)
            #left line
            cv2.line(self.final, (0, int(0.5*height)), (int(0.125*width), height), 255, 10)
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
                cv2.line(self.final, (x, y), (width, height), 255, 10)
                cv2.line(self.final, (0, height), (edge_white_x, edge_white_y), 255, 10)
                
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
        while cap.isOpened():
            ret, self.image = cap.read()
            if ret:
                self.image = self.adjust_gamma(self.image)
                self.height, self.width, _ = self.image.shape
                
                self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

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
