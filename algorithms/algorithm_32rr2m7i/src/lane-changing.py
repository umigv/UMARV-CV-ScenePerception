import numpy as np
import cv2
from ultralytics import YOLO

class ReallyGoodStateMachine:
    def __init__(self):
        # All models are from ARV DropBox
        self.person_model = YOLO('./data/yolov8n.pt')
        self.lines_model = YOLO('./data/best_yolov11_lane_lines.pt')
        self.barrel_model = YOLO('./data/obstacles.pt')
        #

        # these two captures are 1 : from the google drive #9, 
        # and the other is the mirrored version of the same video
        self.cap = cv2.VideoCapture("data/9 function test pedestrian detection lane change & barrel stop.MP4") 
        self.cap_1 = cv2.VideoCapture("data/mirrored_9.mp4") 
        self.cap_5 = cv2.VideoCapture("data/20260322_172804.mp4") 
        self.cap_ = cv2.VideoCapture("data/20260322_172726.mp4") 
        #

        # Frame counts are for reducing frame rate
        self.frame_count = 0
        self.process_per_frame = 3  
        self.right_to_left = True

        # Values for HSV
        self.white_lower_bound = np.array([0, 0, 180])
        self.white_upper_bound = np.array([179, 103, 255])

        # State constants
        self.state_1 = 1 # Person detection -> waits until a person takes up enough of the screen, then: state 2
        self.state_2 = 2 # Lane Change -> provides waypoints to change lanes until the barrel takes up enough of the screen, then: state 3
        self.state_3 = 3 # Drive forward toward barrel until we are close enough

        #starts in looking for people state
        self.state = self.state_1

        self.entered_sentinel = False
        self.exited_sentinel = False

    # Determines whether a lane change should be from Left->Right or Right->Left
    # Determines this through count of white pixels 
    # More white pixels on side x means leaving side x to go to lane on other side of the screen
    # ex: (less white on right side : lane change Right->Left)
    # True means 
    def set_right_or_left(self):
        ret, img = self.cap.read()
        results = self.lines_model(img)
        full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            
        height, width = img.shape[:2]
        result = results[0]
        
        if result.masks is not None:
                # Loop through each detected object
                for mask, cls in zip(result.masks.data, result.boxes.cls):
                    if int(cls) == 0:
                        # result.masks.data is usually lower resolution, 
                        # we convert to numpy and resize to match original image
                        m = mask.cpu().numpy()
                        m = cv2.resize(m, (img.shape[1], img.shape[0]))
                        
                        # Add this object's pixels to our full mask
                        full_mask[m > 0.5] = 255        
        


        mid = width // 2

        mask_l = full_mask[:, 0:mid]
        mask_r = full_mask[:, mid:width]
        
        white_pixel_l = cv2.countNonZero(mask_l)
        white_pixel_r = cv2.countNonZero(mask_r)
        
        if white_pixel_l < white_pixel_r:
            print('right->left lane change')
            return False
        print('left->right lane change')
        return True
    
    def get_mask(self, model, img, mode="person"):
        results = model(img, classes = 0)
        result = results[0]
        m = np.zeros(img.shape[:2], dtype=np.uint8)
        label = np.zeros(img.shape[:2], dtype=np.uint8)
        if(mode == "person"):
            results = self.person_model(img)
            py2 = 0
            for results in results:
                boxes = results.boxes.xyxy.tolist()
                confidences = results.boxes.conf.tolist()
                class_ids = results.boxes.cls.tolist()

                for box, confidence, class_id in zip(boxes, confidences, class_ids):
                    # class 0 is person (built in) and adjust confidence as needed
                    px1, py1, px2, py2 = map(int, box)
                    m[py1:py2, px1:px2] = 255
                    label[py1:py2, px1:px2] = 1 # 3 indicates person class
            return m, label
                        
        else:
            if result.masks is not None:
                # Loop through each detected object
                for mask, cls in zip(result.masks.data, result.boxes.cls):
                    if int(cls) == 0:
                        # result.masks.data is usually lower resolution, 
                        # we convert to numpy and resize to match original image
                        l = cls.cpu().numpy()
                        m1 = cv2.resize(mask.cpu().numpy(), (img.shape[1], img.shape[0]))
                        m = cv2.bitwise_or(m, m1)
                        label = cv2.bitwise_or(label, l)
                        
                return m, label
        return m, label

    #Changes Lanes
    def change_lanes(self, img, y_waypoint, prev_x):
        
    
        full_mask1, lines_label = self.get_mask(self.lines_model, img, mode="lines")
        full_mask2, barrel_label = self.get_mask(self.barrel_model, img, mode="barrel")
        full_mask3, person_label = self.get_mask(self.person_model, img, mode="person")
        
        
        full_mask = cv2.bitwise_or(full_mask1, full_mask2)
        full_mask = cv2.bitwise_or(full_mask, full_mask3)

        # lines - 1, barrel - 2, person - 3 (for visualization purposes)
        # full_label = lines_label + barrel_label*2 + person_label*3
        # print(full_label)
        
        full_img = np.zeros(img.shape[:2], dtype=np.uint8)
        full_img[full_mask > 0.5] = 255 
            
        lanes_mask = full_mask1
        lanes_img = np.zeros(img.shape[:2], dtype=np.uint8)
        lanes_img[lanes_mask > 0.5] = 255 

        if (self.right_to_left):
            x  = self.find_waypoint_left(y_waypoint, lanes_img, prev_x )
        else :
            x = self.find_waypoint_right(y_waypoint, lanes_img, prev_x)
        done_ = False

        width = img.shape[1]
        height, width = img.shape[:2]
        if (x > (width * (0.8))) and (x < (width - 150)):
            # Look for barrel being big enough = at barrel
            barrel_results = full_mask2
            for result in barrel_results:
                boxes = result.boxes.xyxy.tolist()
                confidences = result.boxes.conf.tolist()
                class_ids = result.boxes.cls.tolist()

                for box, confidence, class_id in zip(boxes, confidences, class_ids):
                    BARREL_ID = 0
                    px1, py1, px2, py2 = map(int, box)
                    if class_id == BARREL_ID and confidence > 0.7:
                        
                        print("BARREL")
                        height, width = img.shape[:2]
                        size_barrel = (px2-px1)/(width/3)
                        cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 0), 2)
                        if size_barrel > 0.3:
                            print("barrel within range")
                            done_ = True
        
        
        return done_, x, full_img

    # Finds pedestrian in lane and plots the pedestrian box as well as returning whether they are in range
    #
    def sees_pedestrian_in_lane(self, img):
        
        results = self.person_model(img)
        py2 = 0

        for results in results:
            boxes = results.boxes.xyxy.tolist()
            confidences = results.boxes.conf.tolist()
            class_ids = results.boxes.cls.tolist()

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                # class 0 is person (built in) and adjust confidence as needed
                px1, py1, px2, py2 = map(int, box)
                if class_id == 0 and confidence > 0.7:
                    height, width = img.shape[:2]
                    range = int(width/100)
                    size_person = (px2-px1)/width
                    cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 0), 2)
                    cv2.waitKey(1)    
                    if size_person > 0.12:
                        print("person within range")
                        return True, (py2 + range), px1
                    else:
                        return False, (py2 + range), px1
                    

                
        return False, py2, px1

    def at_barrel(self, capture, img):
        # Placeholder logic
        return False
    
    def add_waypoint(self, y, img, x):
        center = (x, y)
        radius = 25
        color = [255,100,0]
        cv2.circle(img, center, radius, color, thickness=3, lineType=8, shift=0)
        cv2.imshow("waypoint",img)

    def find_waypoint_right(self, y_in, img, prev_x):
        height, width = img.shape
        SENTINEL = -100
        x = SENTINEL       
        spacing = 10
        img_slice = img[y_in - spacing : y_in + spacing, :]

        y_values , x_values = np.where(img_slice == 255)

        if(x_values.size > 0):
            if(np.max(x_values) > int(width/3)):
                x = np.max(x_values)
                return int(x - (width * 3/8))
            

        if(x == SENTINEL and not self.exited_sentinel):
                self.entered_sentinel = True
                x = int (width * (0.75))
                return x
        

        if(self.entered_sentinel):
            self.exited_sentinel = True
        if self.exited_sentinel:
            return int(prev_x )
        # - (width/3)

        

    def find_waypoint_left(self, y_in, img, prev_x):
        height, width = img.shape
        SENTINEL = -100
        x = SENTINEL     
        spacing = 10
        img_slice = img[y_in - spacing : y_in + spacing, :]

        y_values , x_values = np.where(img_slice == 255)

        if(x_values.size > 0):
            if(np.min(x_values) < int(width * 2/3)):
                x = np.min(x_values)
                if(self.entered_sentinel):
                    self.exited_sentinel = True
                return int(x + (width * 3/8))
       
        # Mirror: If nothing found, default to the left-side equivalent (30%)
        if x == SENTINEL and not self.exited_sentinel:
            self.entered_sentinel = True
            x = int(width * 0.25)
            return x
        
        # Mirror: Instead of subtracting 600 (moving left), 
        # add 600 to move right toward the center
        
        if self.exited_sentinel:
            return int(prev_x)
        #  + (width/3)
        

    def run(self):
        running = True
        
        one_waypoint_placed = False
        while(running and self.cap.isOpened()):
            # read frames
            ret, img = self.cap.read()
            if not ret:
                break
            height, width = img.shape[:2]
            # if(self.right_to_left):
            #     prev_x = int(width * 3/4)
            # else:
            #     prev_x = int(width/4)
            prev_x = int(width/2)
            self.frame_count += 1
        
            if self.frame_count % self.process_per_frame != 0:
                continue

            # State Logic
            if(self.state == self.state_1):
                see_pedestrian, y_waypoint, x_waypoint = self.sees_pedestrian_in_lane(img)
                self.add_waypoint(y_waypoint, img, x_waypoint)
                if(see_pedestrian):
                    print("PERSON DETECTED")
                    self.state = self.state_2
                    print(self.state)
                    

             
            elif(self.state == self.state_2):
                
                if(not one_waypoint_placed):
                    self.add_waypoint(y_waypoint,img, x_waypoint)
                    one_waypoint_placed = True
                done, x_waypoint, full_mask = self.change_lanes( img, y_waypoint,prev_x)
                self.add_waypoint(y_waypoint,img, x_waypoint)
                print(f"x_waypoint : {x_waypoint}")
                prev_x = x_waypoint
                cv2.imshow("withwaypoint", full_mask)

                if(done):
                    self.state = 3
                    
            elif(self.state == self.state_3):
                if(self.at_barrel(self.cap, img)):
                    running = False
                    print("AT BARREL")     
            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cap.release()
                cv2.destroyAllWindows()
                break



machine = ReallyGoodStateMachine()
machine.right_to_left = machine.set_right_or_left()
machine.run()