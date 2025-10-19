import dropbox
import json
import os
import sys
import re
import glob
import copy
import cv2
import matplotlib.pyplot as plt
from getpass import getpass
from tqdm.notebook import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

def lane_detector(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    white_threshold = 200
    _, white_mask = cv2.threshold(blurred, white_threshold, 255, cv2.THRESH_BINARY)
    
    edges = cv2.Canny(white_mask, 50, 150)
    
    height, width = edges.shape
    roi_vertices = np.array([[(width*0.1, height), (width*0.45, height*0.6),
                             (width*0.55, height*0.6), (width*0.9, height)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, roi_vertices)
    
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=20,
                            minLineLength=20, maxLineGap=50)
    
    mask = np.zeros_like(edges)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), 1, 5)

    return mask

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img