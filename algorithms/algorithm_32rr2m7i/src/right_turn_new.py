import cv2
import numpy as np
from hsv import hsv


class RightTurn:
    def __init__(self):
        pass

    def state_1(self):
        # state1: when we're at the stopping line of the intersection before starting the right turn
        # start forward movement
        pass

    def state_2(self):
        # state2: in the case we can't see yellow dashed line but past stop line
        # start right movement
        pass

    def state_3(self):
        # state3: the case where we're mid-turn and can see the yellow dashed line
        # using cv2 contours, detect and draw temp lane lines
        pass

    def state_machine(self):
        pass

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
    obj = RightTurn()
    obj.run()

if __name__ == "__main__":
    main()