import cv2
import numpy as np


def get_white_mask(img):
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([360, 100, 255])
    return cv2.inRange(img, lower_white, upper_white)

def get_red_mask(img):
    lower_red1 = (0, 75, 30)
    upper_red1 = (10, 255, 255)
    return cv2.inRange(img, lower_red1, upper_red1)

def get_yellow_mask(img):
    lower_yellow = np.array([20, 120, 150])
    upper_yellow = np.array([40, 255, 255])
    return cv2.inRange(img, lower_yellow, upper_yellow)

def get_blue_mask(img):
    lower_blue = np.array([100, 120, 150])
    upper_blue = np.array([140, 255, 255])
    return cv2.inRange(img, lower_blue, upper_blue)

def get_green_mask(img):    
    lower_green = np.array([40, 120, 150])
    upper_green = np.array([80, 255, 255])
    return cv2.inRange(img, lower_green, upper_green)

def get_seg_mask(img):
    image = cv2.imread(img)
    imagehsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = get_white_mask(imagehsv)
    # mask2 = get_red_mask(imagehsv)
    # mask3 = get_blue_mask(imagehsv)
    # mask4 = get_green_mask(imagehsv)
    # mask5 = get_yellow_mask(imagehsv)
    
    # final_mask = cv2.bitwise_or(mask1, mask2)
    # final_mask = cv2.bitwise_or(final_mask, mask3)
    # final_mask = cv2.bitwise_or(final_mask, mask4)
    # final_mask = cv2.bitwise_or(final_mask, mask5)
    
    kernel = np.ones((5, 5), np.uint8) 

    mask = cv2.erode(mask1, kernel, iterations=1) 
    mask = cv2.dilate(mask, kernel, iterations=1)
        
    result = cv2.bitwise_and(image, image, mask=mask)
    
    
    cv2.imshow('image', image)
    cv2.imshow('imagehsv', imagehsv)
    cv2.imshow('mask', mask)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    get_seg_mask('/Users/mgawthro/Desktop/UMARV/UMARV-CV-ScenePerception/algorithms/algorithm_32rr2m7i/initial_work/data/000003.jpg')