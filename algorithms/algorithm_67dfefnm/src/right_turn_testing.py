# 
import cv2
import numpy as np

def update_mask(image, hsv_image):
    #defining the ranges for HSV values
    yellow_lower_bound = np.array([21, 87, 0])
    yellow_upper_bound = np.array([27, 255, 255])
    
    
    mask = cv2.inRange(hsv_image, yellow_lower_bound, yellow_upper_bound) # Return a mask of HSV values within the range we specified
    
    #some post processing stuff to help it look smoother
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    
    result = cv2.bitwise_and(image, image, mask=mask)
    resized_mask = cv2.resize(mask, (640, 480))
    cv2.imshow("result", resized_mask)
    cv2.waitKey(0)
    
    
    
# Make main function that gets the image path and passes it into update mask as an HSV converted image
# use cv2.imread(filename) then put the output into cv2.cvtcolor(image, cv2.RBGTOHSV)





image = cv2.imread('yellow_dashed_center_2.png')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

update_mask(image, hsv_image)

# start_point = 
# end_point =
# color = 
# thickness = 


# cv2.line(image, start_point, end_point, color, thickness)