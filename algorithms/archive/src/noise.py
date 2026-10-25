import cv2 as cv
import numpy as np
import random

# Define function to load an image, add noise to the image, and save a noisy copy (a bit slow)
# TODO: See how Gaussian Blur affects things
def addNoise(img_loc, write_loc):
    img = cv.imread(img_loc)
    """
    For more relevant noise, create random gaussian blobs across the whole image
    """
    random.seed(200)

    # TODO: These settings are too high, they cause the model itself to fail
    blob_prob = 0.0005 #per pixel
    noise_magnitude_normalized = 0.3 
    max_blob_width_normalized = 0.5
    #max_blob_height_normalized = 0.5

    blob_centers = []
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if random.random() <= blob_prob:
                blob_centers.append((x,y))
    print(len(blob_centers))
    noisy_img = img.copy()
    for cx,cy in blob_centers:
        blob_width = np.ceil(random.random()*max_blob_width_normalized*img.shape[1])
        #blob_height = math.ceil(random.random()*max_blob_height_normalized*img.shape[0])
        blob_height = blob_width
        noise_magnitude = noise_magnitude_normalized*255
        X = np.linspace(-3, 3, blob_width)[None, :]
        Y = np.linspace(-3, 3, blob_height)[:, None]
        gaussian_blob = np.exp(-0.5*X**2) * np.exp(-0.5*Y**2)
        gaussian_blob = noise_magnitude * (gaussian_blob - gaussian_blob.min()) / gaussian_blob.max()

        # Add the blob
        for gx in range(blob_width):
            x_idx = cx + gx - int(blob_width/2)
            if x_idx < 0 or x_idx >= img.shape[1]: 
                continue
            for gy in range(blob_height):
                y_idx = cy + gy - int(blob_height/2)
                if y_idx < 0 or y_idx >= img.shape[0]:
                    continue
                noisy_img[y_idx, x_idx] += int(gaussian_blob[gy, gx])
    noisy_img = np.clip(noisy_img, 0, 255)
    cv.imwrite(write_loc, noisy_img)

#  This function creates a noisy copy of a binary mask
def addNoiseMask(mask, sigma):
    noise = np.random.normal(0,sigma, mask.shape)
    noisy_mask = np.where(mask + noise >= 0.5, 1, 0)
    return noisy_mask.astype(np.uint8)