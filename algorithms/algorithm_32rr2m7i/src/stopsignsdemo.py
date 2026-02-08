import cv2
import pytesseract
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

model = YOLO('./data/stopsigns.pt')

output_dir = 'processed_images'
input_dir = 'input_images'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(input_dir, exist_ok=True)
count = 0

custom_oem_psm_config = r'--oem 3 --psm 6 --user-words /content/user_words.txt -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

#get images
# image_dir = r'.\images'  
image_dir = r'input_images'  
#image_dir = r'.\stop sign images'
valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')

#words we wanna find
with open('user_words.txt', 'w') as f:
    f.write("IGVC\nsoup\nSTOP")


for idx, filename in enumerate(os.listdir(image_dir)):
  
    count+=1
    if filename.lower().endswith(valid_extensions):
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Could not read image {filename}")

        results = model(image)
        texts = []

        for result in results:
            boxes = result.boxes.xyxy.tolist()
            confidences = result.boxes.conf.tolist()
            class_ids = result.boxes.cls.tolist()

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            #confidence > 80%
            # print(confidence)
            if confidence > 0.8:
                x1, y1, x2, y2 = map(int, box)

                #crop it
                cropped_image = image[y1:y2, x1:x2]
                # new idea

                # CROP_FACTOR_Y = 0.2 * (y2 - y1)
                # CROP_FACTOR_X = 0.04 * (x2 - x1)

                # y1 += int(CROP_FACTOR_Y)
                # y2 -= int(CROP_FACTOR_Y)
                # x1 += int(CROP_FACTOR_X)
                # x2 -= int(CROP_FACTOR_X)
                # cropped_image = image[y1:y2, x1:x2]

                plt.figure(figsize=(15, 8))  

                #original image
                plt.subplot(2, 4, 1)  
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.title("0. Original Image")
                plt.axis('off')

                #yolo det
                plt.subplot(2, 4, 2)
                plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                plt.title("1. YOLO Detection")
                plt.axis('off')

                #red mask
                hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
                lower_red1 = np.array([0, 70, 50])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([160, 70, 50])
                upper_red2 = np.array([180, 255, 255])
                # temporarily allow all colors
                # lower_red2 = np.array([0, 0, 0])
                # upper_red2 = np.array([255, 255, 255])

                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                mask = cv2.bitwise_or(mask1, mask2)

                plt.subplot(2, 4, 3)
                plt.imshow(mask, cmap='gray')
                plt.title("2. Red Color Mask")
                plt.axis('off')

                #red-cropped
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                red_cropped = cropped_image.copy()

                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    red_cropped = cropped_image[y:y+h, x:x+w]

                plt.subplot(2, 4, 4)
                if len(red_cropped.shape) == 3:
                    plt.imshow(cv2.cvtColor(red_cropped, cv2.COLOR_BGR2RGB))
                else:
                    plt.imshow(red_cropped, cmap='gray')
                plt.title("3. Red Region Cropped")
                plt.axis('off')

                #ratio-based crop
                if red_cropped.size > 0:
                    h, w = red_cropped.shape[:2]
                    realy1 = int(h / 4)
                    realy2 = int(h - (h / 4))
                    # realx1 = int(w / 14)
                    # realx2 = int(w - (w / 14))
                    realx1 = int(w / 23)
                    realx2 = int(w - (w / 23))

                    if realy2 > realy1 and realx2 > realx1:
                        final_crop = red_cropped[realy1:realy2, realx1:realx2]
                    else:
                        final_crop = red_cropped
                else:
                    final_crop = cropped_image

                plt.subplot(2, 4, 5)
                if len(final_crop.shape) == 3:
                    plt.imshow(cv2.cvtColor(final_crop, cv2.COLOR_BGR2RGB))
                else:
                    plt.imshow(final_crop, cmap='gray')
                plt.title("4. Ratio-Based Crop")
                plt.axis('off')

                #inverted grayscale
                if len(final_crop.shape) == 3:
                    gray_image = cv2.cvtColor(final_crop, cv2.COLOR_BGR2GRAY)
                else:
                    gray_image = final_crop.copy()

                inverted_image = cv2.bitwise_not(gray_image)
                plt.subplot(2, 4, 6)
                plt.imshow(inverted_image, cmap='gray')
                plt.title("5. Inverted Grayscale")
                plt.axis('off')

                #thresholded image
                _, thresholded = cv2.threshold(inverted_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                plt.subplot(2, 4, 7)
                plt.imshow(thresholded, cmap='gray')
                plt.title("6. Thresholded Result")
                plt.axis('off')

                #OCR-detected text and confidence
                text = pytesseract.image_to_string(thresholded, config=custom_oem_psm_config).strip()
                plt.subplot(2, 4, 8)
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
                plt.text(10, 30, f"Detected Text: {text}\nConfidence: {confidence:.2f}", fontsize=10, color='red', backgroundcolor='white')
                plt.axis('off')

                plt.tight_layout()
                plt.pause(15)
                    
                  # Pause for 2 seconds before moving to the next image
                plt.close()

                
                print(f"\nProcessing {filename}:")
                print(f"Detected text: {text}") # make text lower for uniformity
                print(f"Confidence: {confidence:.2f}")
                print(f"Class ID: {class_id}")

print("\nProcessing complete!")


