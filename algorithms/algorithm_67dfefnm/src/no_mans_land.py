import cv2
import numpy as np

from hsv import hsv

class NoMansLand:
    def __init__(self):
        self.hsv_obj = None
        self.image = None

        self.ramp_mask = None
        self.ramp_mode = False

        self.final = None

    def update_mask(self):
        #defining the ranges for HSV values
        self.final, dict = self.hsv_obj.get_mask(self.image)
        
        self.ramp_mask = dict["ramp_color"]
        
        # self.past_stop_line()
        
        # if(self.done == False):
        # self.last_diff_y = self.find_left_most_lane()
        # else:
        #     self.draw_trapazoid()
        #     self.centroid = (self.width//2, 40)
        # self.find_center_of_lane()
        # cv2.namedWindow("mask", self.final)
        # cv2.imshow("mask", self.final)
        final_bgr = cv2.cvtColor(self.final, cv2.COLOR_GRAY2BGR)
        combined = np.vstack((self.image, final_bgr))
        # cv2.imshow("mask", combined)

    def grab_ramp_corners(self, best_cnt):
        min_x = self.width - 1
        max_x = 0
        min_y = self.height - 1
        max_y = 0

        for point in best_cnt:
            if point[0, 0] > max_x:
                max_x = point[0, 0]
            if point[0, 0] < min_x:
                min_x = point[0, 0]

            if point[0, 1] > max_y:
                max_y = point[0, 1]
            if point[0, 1] < min_y:
                min_y = point[0, 1]

        # cv2.circle(self.final, (min_x, max_y), 5, 128, -1)
        # cv2.circle(self.final, (max_x, max_y), 5, 128, -1)

        return (min_x, max_y), (max_x, max_y), (min_x, min_y), (max_x, min_y) #bottom left, bottom right, top left, top right
        
    def run(self):
        cap = cv2.VideoCapture('data/ramp.MOV') # 0 for webcam # 1,2 for external cameras
        self.hsv_obj = hsv("data/ramp.MOV")

        # self.hsv_obj.tune("ramp_color")
                    
        while cap.isOpened():
            ret, self.image = cap.read()
            self.height, self.width, _ = self.image.shape

            if ret:
                self.update_mask()
                cnts, _ = cv2.findContours(self.ramp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                self.ramp_mode = False

                bottom_left = (None, None)
                bottom_right = (None, None)
                top_left = (None, None)
                top_right = (None, None)

                min_area = 1000
                for cnt in cnts:
                    if cv2.contourArea(cnt) > min_area:
                        self.ramp_mode = True
                        bottom_left, bottom_right, top_left, top_right = self.grab_ramp_corners(cnt)
                        
                        cv2.line(self.final, bottom_left, (0, self.height), 255, 10)
                        cv2.line(self.final, bottom_right, (self.width, self.height), 255, 10)

                        mid_bottom = ((bottom_left[0] + bottom_right[0]) // 2, (bottom_left[1] + bottom_right[1]) // 2)
                        mid_top = ((top_left[0] + top_right[0])// 2, (top_left[1] + top_right[1]) // 2)
                        cv2.circle(self.final, mid_bottom, 5, 128, -1) #radius, color, ?
                        cv2.circle(self.final, mid_top, 5, 128, -1)

                if self.ramp_mode:
                    print("ramp mode")
                else:
                    print("not ramp mode")

                cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
                cv2.imshow("Video", self.image)
                cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
                cv2.imshow("mask", self.final)

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
    
    def detect_ramp(color_range):
        pass



def main():
    obj = NoMansLand()
    obj.run()
    obj.detect_ramp(123) #fill with color range (green)

if __name__ == "__main__":
    main()