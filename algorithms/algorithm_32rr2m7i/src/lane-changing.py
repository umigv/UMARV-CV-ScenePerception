import numpy as np
import cv2
from ultralytics import YOLO

class SolidStateMachine:
    def __init__(self):
        # Keep original variable names exactly as provided
        self.person_model = YOLO('./data/yolov8n.pt')
        self.lines_model = YOLO('./data/best_yolov11_lane_lines.pt')
        self.barrel_model = YOLO('./data/obstacles.pt')
        # these two captures are 1 : from the google drive #9, 
        # and the other is the mirrored version of the same video
        # self.cap = cv2.VideoCapture("data/9 function test pedestrian detection lane change & barrel stop.MP4") 
        self.cap = cv2.VideoCapture("data/mirrored_9.mp4") 
        self.frame_count = 0
        self.process_per_frame = 3
        self.right_to_left = True

        # Values for HSV
        self.white_lower_bound = np.array([0, 0, 180])
        self.white_upper_bound = np.array([179, 103, 255])

        # State constants
        self.state_1 = 1
        self.state_2 = 2
        self.state_3 = 3
        self.state = self.state_1
        class_ids = list(self.barrel_model.names.keys())
        # print(class_ids)

    def set_right_to_left(self):
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
            print('right lane change')
            return False
        print('left lane change')
        return True
    

    def change_lanes(self, capture, img, y_waypoint, prev_x):
        right_lane_change = True
        if(right_lane_change):
            results = self.lines_model(img)
        
            # Create an empty black mask the same size as the image
            full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            
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
        
        if (not self.right_to_left):
            x = self.find_x_state_2_Right(y_waypoint, full_mask, prev_x)
        else :
            x = self.find_x_state_2_Left(y_waypoint, full_mask, prev_x)
        # cv2.imshow("img", img)
        done_ = False

        width = img.shape[1]
        height, width = img.shape[:2]
        if (x > width * (0.8)) and (x < (width - 150)):
            # Look for barrel being big enough = at barrel
            barrel_results = self.barrel_model(img)
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
        
        
        return done_, x, full_mask

    def sees_pedestrian_in_lane(self, capture, img):
        
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
                    
                    # print("PERSON")
                    height, width = img.shape[:2]
                    size_person = (px2-px1)/width
                    
                    
                    cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 0), 2)
                    # cv2.imshow("label", img)
                    cv2.waitKey(1)    
                    if size_person > 0.12:
                        print("person within range")
                        return True, py2, px1
                    else:
                        return False, py2, px1
                    

                
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

    # def find_x_state_1( px1){
    #      return px1
    # }
    def find_x_state_2_Right(self, py2, img, prev_x):
        height, width = img.shape
        x = 1       
        row = img[py2, :]
        for i in range(width - 1, -1, -1):
            if row[i] == 255:
                x = i
                print("found right")
                break
        if(x == 1):
             x = int (width * (0.7))
             return x
        
        return x - 600
    
    # def find_x_state_2_Left(self, py2, img, prev_x):
    #     height, width = img.shape
    #     x = width - 1    
    #     row = img[py2, :]
    #     for i in range(0, width-1,  1):
    #         if row[i] == 255:
    #             x = i
    #             print("found right")
    #             break
    #     if(x == width - 1):
    #          x = int (width * (0.3))
    #          return x
        
    #     return x - 600
    
    def find_x_state_2_Left(self, py2, img, prev_x):
        height, width = img.shape
        # Keep the flag consistent with your original code
        x = 1       
        row = img[py2, :]
        
        # Mirror: Scan from left (0) to right (width)
        for i in range(0, width):
            if row[i] == 255:
                x = i
                print("found left")
                break
                
        # Mirror: If nothing found, default to the left-side equivalent (30%)
        if x == 1:
            x = int(width * 0.3)
            return x
        
        # Mirror: Instead of subtracting 600 (moving left), 
        # add 600 to move right toward the center
        return (x + 600)

    def run(self):
        running = True
        
        one_waypoint_placed = False
        while(running and self.cap.isOpened()):
            # read frames
            ret, img = self.cap.read()
            height, width = img.shape[:2]
            if(self.right_to_left):
                prev_x = int(width * 3/4)
            else:
                prev_x = int(width/4)
            
            if not ret:
                break
            
            self.frame_count += 1
        
            if self.frame_count % self.process_per_frame != 0:
                continue

            # State Logic
            if(self.state == self.state_1):
                see_pedestrian, y_waypoint, x_waypoint = self.sees_pedestrian_in_lane(self.cap, img)
                self.add_waypoint(y_waypoint, img, x_waypoint)
                if(see_pedestrian):
                    print("PERSON DETECTED")
                    self.state = self.state_2
                    print(self.state)
                    cv2.destroyAllWindows()

             
            elif(self.state == self.state_2):
                if(not one_waypoint_placed):
                    self.add_waypoint(y_waypoint,img, x_waypoint)
                    one_waypoint_placed = True
                done, x_waypoint, full_mask = self.change_lanes(self.cap, img, y_waypoint,prev_x)
                self.add_waypoint(y_waypoint,img, x_waypoint)
                print(f"x_waypoint : {x_waypoint}")
                prev_x = x_waypoint
                cv2.imshow("withwaypoint", full_mask)
                cv2.waitKey(1)

                if(done):
                    self.state = 3
                    
            elif(self.state == self.state_3):
                if(self.at_barrel(self.cap, img)):
                    running = False
                    print("AT BARREL")     
            
            
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     self.cap.release()
            #     cv2.destroyAllWindows()
            #     break

machine = SolidStateMachine()
machine.right_to_left = machine.set_right_to_left()
machine.run()