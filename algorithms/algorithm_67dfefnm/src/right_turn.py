import cv2
import numpy as np

def update_mask(image, hsv_image):
    #defining the ranges for HSV values
    yellow_lower_bound = np.array([12, 95, 0])
    yellow_upper_bound = np.array([32, 255, 255])
    white_lower_bound = np.array([0, 0, 180])
    white_upper_bound = np.array([179, 103, 255])
    
    
    white_mask = cv2.inRange(hsv_image, white_lower_bound, white_upper_bound)
    yellow_mask = cv2.inRange(hsv_image, yellow_lower_bound, yellow_upper_bound) # Return a mask of HSV values within the range we specified
    
    white_mask = cv2.resize(white_mask, (image.shape[1], image.shape[0]))
    yellow_mask = cv2.resize(yellow_mask, (image.shape[1], image.shape[0]))

    #some post processing stuff to help it look smoother
    white_mask = cv2.erode(white_mask, None, iterations=2)
    white_mask = cv2.dilate(white_mask, None, iterations=2)

    yellow_mask = cv2.erode(yellow_mask, None, iterations=2)
    yellow_mask = cv2.dilate(yellow_mask, None, iterations=2)
    
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    
    find_left_most_lane(yellow_mask, mask)
    
    # result = cv2.bitwise_and(image, image, mask=mask) #applies the mask to the base image so only masked parts of the image are shown
    resized_mask = cv2.resize(mask, (640, 480))
    # cv2.imshow("result", result)
    cv2.imshow("mask", resized_mask)
    
    
    
def find_left_most_lane(mask, final):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 200 # Adjust based on noise size
    hsv_lanes = np.zeros_like(mask) #new occupancy grid that is blank
    
    min_x = float("inf")
    best_cnt = None
    max_y = 0
    
    for cnt in contours: # looping through contours
        if cv2.contourArea(cnt) > min_area:
            #find left most lane
            print(cnt[0, 0, 0]) # x-values of each cnt
            print(cnt[0, 0, 1]) # y-values of each cnt
            
            if cnt[0, 0, 1] > max_y:
                max_y = cnt[0, 0, 1]
                min_x = cnt[0, 0, 0]
                best_cnt = (min_x, max_y)
            
            cv2.drawContours(hsv_lanes, [cnt], -1, 255, thickness=cv2.FILLED)
            cv2.circle(final, best_cnt, 50, 255, -1)
            
            cv2.line(final, (0, best_cnt[1]), (0, mask.shape[0]), 255, 10)
            cv2.line(final, (best_cnt[0], best_cnt[1]), (0, best_cnt[1]), 255, 10)
    
# Make main function that gets the image path and passes it into update mask as an HSV converted image
# use cv2.imread(filename) then put the output into cv2.cvtcolor(image, cv2.RBGTOHSV)


cap = cv2.VideoCapture("data/right_turn.mp4")
while cap.isOpened():
    ret, image = cap.read()
    if ret:
        height, width, _ = image.shape
        right_half_vid = image[:, width // 2:]
        hsv_image = cv2.cvtColor(right_half_vid, cv2.COLOR_BGR2HSV)
       
        update_mask(image, hsv_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break






# 
# start_point = 
# end_point =
# color = 
# thickness = 


# cv2.line(image, start_point, end_point, color, thickness)