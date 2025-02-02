import cv2 as cv
from matplotlib.path import Path
import numpy as np
import os
from roboflow import Roboflow
from ultralytics import YOLO

def download_dataset():
    # Download existing dataset
    # TODO ADD ROBOFLOW CODE HERE
    pass

def load_and_run_models():
    # Load models
    HOME = os.getcwd()
    MODEL_LOC = HOME + "/datasets/april9120sLLO.pt"
    model = YOLO(MODEL_LOC)

    # Load test case and run model on it
    TEST_CASE = "youtube-9_jpg.rf.6129ca3c07ffb4cc8ed82fb6e09ab3eb"
    TEST_IMG_LOC = HOME + "/Drivable-area-model-9/valid/images/" + TEST_CASE + ".jpg"
    results = model.predict(TEST_IMG_LOC)
    img = cv.imread(TEST_IMG_LOC)
    img_uint = np.uint8(img)
    true_mask = create_true_mask(TEST_CASE, img_uint)
    return results, img_uint, true_mask


def create_true_mask(test_case, img):
    HOME = os.getcwd()
    TEST_IMG_LABELS = HOME + "/Drivable-area-model-9/valid/labels/" + test_case + ".txt" 
    raw = None
    with open(TEST_IMG_LABELS) as file:
        for line in file:
            raw = line.split()[1:]
    convex_hull = np.array([float(x) for x in raw]).reshape(int(len(raw)/2),2)
    convex_hull[:,0] = convex_hull[:,0]*img.shape[1]
    convex_hull[:,1] = convex_hull[:,1]*img.shape[0]
    convex_hull = convex_hull.astype(np.uint32)
    x,y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T 
    grid = np.reshape(Path(convex_hull).contains_points(points).T, img.shape[:-1])
    return grid.astype(np.uint8)


