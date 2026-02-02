import cv2
import numpy as np

from hsv import hsv

class NoMansLand:
    def __init__(self):
        pass

    def update_mask(self):
        #defining the ranges for HSV values
        self.final, dict = self.hsv_obj.get_mask(self.image)
        
        self.white_mask = dict["white"]
        self.yellow_mask = dict["yellow"]
        
        self.past_stop_line()
        
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
        



def run(self):
    cap = cv2.VideoCapture('data/trimmed_ramp.mp4') # 0 for webcam # 1,2 for external cameras
                
    while cap.isOpened():
        ret, self.image = cap.read()
        if ret:
            # add function logic here

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
    obj = NoMansLand()
    obj.run()

if __name__ == "__main__":
    main()