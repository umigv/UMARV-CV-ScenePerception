import cv2 as cv
import numpy as np

from dbscan_processing import apply_DBSCAN
from utils import applyMask, createDifferenceImg, erodeAndDilate
from noise import addNoiseMask
from roboflow_processing import load_and_run_models

results, img, true_mask = load_and_run_models()
true_img = applyMask(img, true_mask)

include_erode_dilate = True # Pretty fast
include_dbscan = False # Takes a while with the current approach

eps = 4 
min_samples = 20 
sigma_vals = np.linspace(0,0.4,5)
for r in results:
    noiseless_mask = np.uint8(r.masks.data[0].numpy())
    stacked_imgs = [] 
    for sigma in sigma_vals:
        images = []
        images.append(true_img)
        noisy_mask = addNoiseMask(noiseless_mask, sigma)
        mask_noDBSCAN = noisy_mask
        new_img_noDBSCAN = applyMask(img, mask_noDBSCAN)
        diff_true_noDBSCAN_img = createDifferenceImg(mask_noDBSCAN, true_mask)
        images.append(new_img_noDBSCAN)
        images.append(diff_true_noDBSCAN_img)
        if include_erode_dilate:
            basic_mask = erodeAndDilate(noisy_mask)
            new_img_basic = applyMask(img, basic_mask)
            diff_true_erodedilate_img = createDifferenceImg(basic_mask, true_mask)
            images.append(new_img_basic)
            images.append(diff_true_erodedilate_img)
        if include_dbscan:
            mask_DBSCAN = apply_DBSCAN(noisy_mask, eps, min_samples)
            new_img_DBSCAN = applyMask(img, mask_DBSCAN)
            diff_true_DBSCAN_img = createDifferenceImg(mask_DBSCAN, true_mask)
            images.append(new_img_DBSCAN)
            images.append(diff_true_DBSCAN_img)
        stacked_raw = np.hstack(images)
        stacked = cv.resize(stacked_raw, (int(stacked_raw.shape[1]*0.2), int(stacked_raw.shape[0]*0.2)))
        stacked_imgs.append(stacked)
    final_img = np.vstack(tuple(stacked_imgs))
    if include_dbscan and include_erode_dilate:
        title = "True/Noisy/DiffNoisy/ErodeDilate/DiffErodeDilate/DBSCAN/DiffDBSCAN - Sigma (Noise) from " + str(sigma_vals[0]) + "-" + str(sigma_vals[-1])
    elif include_dbscan:
        title = "True/Noisy/DiffNoisy/DBSCAN/DiffDBSCAN - Sigma (Noise) from " + str(sigma_vals[0]) + "-" + str(sigma_vals[-1])
    elif include_erode_dilate:
        title = "True/Noisy/DiffNoisy/ErodeDilate/DiffErodeDilate - Sigma (Noise) from " + str(sigma_vals[0]) + "-" + str(sigma_vals[-1])
    else:
        title = "True/Noisy/DiffNoisy - Sigma (Noise) from " + str(sigma_vals[0]) + "-" + str(sigma_vals[-1])
    cv.imshow(title, final_img)
    cv.waitKey(0)  