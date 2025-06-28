import cv2
import numpy as np

def find_slope(cur_x, cur_y, edge_white_x, edge_white_y):
    diff_y = (edge_white_y - cur_y)
    diff_x = (edge_white_x - cur_x)

    return diff_x, diff_y

def update_mask(image, hsv_image):
    #defining the ranges for HSV values
    yellow_lower_bound = np.array([0, 39, 227])
    yellow_upper_bound = np.array([93, 255, 255])
    white_lower_bound = np.array([0, 0, 201])
    white_upper_bound = np.array([179, 70, 255])
    
    
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
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 200 # Adjust based on noise size
    final_mask = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)
    
    find_left_most_lane(yellow_mask, white_mask, mask)
    
    
    
    
    # result = cv2.bitwise_and(image, image, mask=mask) #applies the mask to the base image so only masked parts of the image are shown
    # resized_mask = cv2.resize(final_mask, (640, 480))
    # cv2.imshow("result", result)
    
    cv2.imshow("mask", mask)
    
    

    
    
def find_left_most_lane(mask, white_mask, final):
    assert(mask.shape == white_mask.shape == final.shape)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 200 # Adjust based on noise size
    hsv_lanes = np.zeros_like(mask) # New occupancy grid that is blank
    height, width = white_mask.shape
    
    min_x = float("inf")
    best_cnt = None
    max_y = 0
    
    for cnt in contours: # Looping through contours
        if cv2.contourArea(cnt) > min_area:
            # Find left most lane
            if cnt[0, 0, 1] > max_y and cnt[0, 0, 0] < width // 2 and cnt[0, 0, 1] < height // 2:
                max_y = cnt[0, 0, 1]
                best_cnt = cnt
            
            cv2.drawContours(hsv_lanes, [cnt], -1, 255, thickness=cv2.FILLED)
    
    max_y = 0
    x, y = None, None
    if best_cnt is not None:
        for point in best_cnt:
            if point[0][1] > max_y:
                y = point[0][1]
                x = point[0][0]
            
    diff_x, diff_y = None, None

    if y is not None:
        # print(x, y)
        
        # cv2.circle(final, (x, y), 100, 255, -1)
        
        edge_white_x = x
        edge_white_y = y
        
        x -= 150
        x = max(0, x)
        
        
        
        while y < height and white_mask[y, x] != 255:
            y += 1
            cv2.circle(final, (x, y), 50, 255, -1)
        
        diff_x, diff_y = find_slope(x, y, edge_white_x, edge_white_y)            
            
        diff_x //= 10
        diff_y //= 10
        x += diff_x * 2
        y += diff_y * 2
        
       
        
        while x < width and y < height and white_mask[y, x] == 0:
            cv2.circle(final, (x, y), 10, 255, -1)

            x += diff_x #* 2
            y += diff_y #* 2
        
        cv2.line(final, (x, y), (width, height), 255, 10)
        
    
# Make main function that gets the image path and passes it into update mask as an HSV converted image
# use cv2.imread(filename
# ) then put the output into cv2.cvtcolor(image, cv2.RBGTOHSV)


def adjust_gamma(image, gamma=0.4):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    image = cv2.LUT(image, table)
    return image

cap = cv2.VideoCapture('data/trimmed.mov')
while cap.isOpened():
    ret, image = cap.read()
    if ret:
        image = adjust_gamma(image)
        height, width, _ = image.shape
        
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
       
        update_mask(image, hsv_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
