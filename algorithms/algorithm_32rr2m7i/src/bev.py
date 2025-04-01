import cv2
from cv2 import cuda
import numpy as np
from math import radians, cos, floor
from ultralytics import YOLO
import math
import time

lane_model = YOLO("april9120sLLO.pt")
hole_model = YOLO('potholesonly100epochs.pt')

class CameraProperties(object):
    functional_limit = radians(70.0)
    def __init__(self, height, fov_vert, fov_horz, cameraTilt):
        self.height = float(height)
        self.fov_vert = radians(float(fov_vert))
        self.fov_horz = radians(float(fov_horz))
        self.cameraTilt = radians(float(cameraTilt))
        self.bird_src_quad = None
        self.bird_dst_quad = None
        self.matrix = None
        self.maxHeight = None
        self.maxWidth = None
        self.minIndex = None

    def src_quad(self, rows, columns):
        if self.bird_src_quad is None:
            self.bird_src_quad = np.array([[0, rows - 1], [columns - 1, rows - 1], [0, 0], [columns - 1, 0]], dtype = 'float32')
        return self.bird_src_quad

    def dst_quad(self, rows, columns, min_angle, max_angle):
        if self.bird_dst_quad is None:
            fov_offset = self.cameraTilt - self.fov_vert/2.0
            bottom_over_top = cos(max_angle + fov_offset)/cos(min_angle + fov_offset)
            bottom_width = columns*bottom_over_top
            blackEdge_width = (columns - bottom_width)/2
            leftX = blackEdge_width
            rightX = leftX + bottom_width
            self.bird_dst_quad = np.array([[leftX, rows], [rightX, rows], [0, 0], [columns, 0]], dtype = 'float32')
        return self.bird_dst_quad

    def reset(self):
        self.bird_src_quad = None
        self.bird_dst_quad = None
        self.matrix = None
        self.maxHeight = None
        self.maxWidth = None
        self.minIndex = None

    def compute_min_index(self, rows, max_angle):
        self.minIndex = int(rows*(1.0 - max_angle/self.fov_vert))
        return self.minIndex

    def compute_max_angle(self):
        return min(CameraProperties.functional_limit - self.cameraTilt + self.fov_vert/2.0, self.fov_vert)

def getBirdView(image, cp):
    rows, columns = image.shape[:2]
    #print(rows, columns)
    if columns == 1280:
        columns = 1344
    if rows == 720:
        rows = 752
    min_angle = 0.0
    max_angle = cp.compute_max_angle()
    min_index = cp.compute_min_index(rows, max_angle)
    image = image[min_index:, :]
    rows = image.shape[0]

    src_quad = cp.src_quad(rows, columns)
    dst_quad = cp.dst_quad(rows, columns, min_angle, max_angle)
    warped, bottomLeft, bottomRight, topRight, topLeft = perspective(image, src_quad, dst_quad, cp)
    return warped, bottomLeft, bottomRight, topRight, topLeft, cp.maxWidth, cp.maxHeight

def perspective(image, src_quad, dst_quad, cp):
    bottomLeft, bottomRight, topLeft, topRight = dst_quad
    widthA = topRight[0] - topLeft[0]
    widthB = bottomRight[0] - bottomLeft[0]
    maxWidth1 = max(widthA, widthB)
    heightA = bottomLeft[1] - topLeft[1]
    heightB = bottomRight[1] - topRight[1]
    maxHeight1 = max(heightA, heightB)

    matrix1 = cv2.getPerspectiveTransform(src_quad, dst_quad)
    cp.matrix = matrix1
    cp.maxWidth = int(maxWidth1)
    cp.maxHeight = int(maxHeight1)
    print("Matrix")
    print(image.shape)
    warped = cv2.warpPerspective(image, matrix1, (cp.maxWidth, cp.maxHeight))


    return warped, bottomLeft, bottomRight, topRight, topLeft

def apply_custom_color_map(image):
    lut = np.zeros((256, 1, 3), dtype=np.uint8)

    lut[130, 0, :] = [255, 0, 0]  # Red for 2 (now 131)
    lut[127, 0, :] = [0, 0, 255]  # Blue for -1 (now 130)
    lut[129, 0, :] = [0, 255, 0]  # Black for 1 (now 129)
    lut[128, 0, :] = [255, 255, 255]  # White for 0 (now 128)

    # Offset the image values to make them all positive
    image = image + 128

    # Ensure the image is in the correct format
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Map the image values to the custom color map
    image_mapped = cv2.LUT(image, lut)

    return image_mapped


def get_occupancy_grid(frame):
        
        r_lane = lane_model.predict(frame, conf=0.5, device='cpu')[0]
        # Save a sample detection to a file

        save = True
        if save:
            with open("lane_detection.txt", "w") as f:
                f.write(str(r_lane))


        # lane_annotated_frame = r_lane.plot()
        image_width, image_height = frame.shape[1], frame.shape[0]
        
        occupancy_grid = np.zeros((image_height, image_width))
        r_hole = hole_model.predict(frame, conf=0.25, device='cpu')[0]
        time_of_frame = 0
        if r_lane.masks is not None:
            if(len(r_lane.masks.xy) != 0):
                with open("lane_detection_xy.txt", "w") as f:
                    f.write(str(r_lane.masks.xy))
                segment = r_lane.masks.xy[0]
                segment_array = np.array([segment], dtype=np.int32)
                cv2.fillPoly(occupancy_grid, [segment_array], color=(255, 0, 0))
                time_of_frame = time.time()

        if r_hole.boxes is not None:
            for segment in r_hole.boxes.xyxyn:
                x_min, y_min, x_max, y_max = segment
                vertices = np.array([[x_min*image_width, y_min*image_height], 
                                    [x_max*image_width, y_min*image_height], 
                                    [x_max*image_width, y_max*image_height], 
                                    [x_min*image_width, y_max*image_height]], dtype=np.int32)
                cv2.fillPoly(occupancy_grid, [vertices], color=(0, 0, 0))

        buffer_area = np.sum(occupancy_grid)//255
        buffer_time = math.exp(-buffer_area/(image_width*image_height)-0.7)
        return occupancy_grid, buffer_time, time_of_frame


def main():

    # ZED = CameraProperties(54, 68.0, 101.0, 68.0)
    ZED = CameraProperties(64, 68.0, 101.0, 60.0)

    cap = cv2.VideoCapture('IMG_7493.mp4')

    out = None
    
    curr_time = time.time()
    memory_buffer = np.full((1280, 720), 255).astype(np.uint8)
    while True:
        # Read in an image:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        occupancy_grid_display, buffer_time, time_of_frame = get_occupancy_grid(frame)
        occ = occupancy_grid_display.astype(np.uint8)
        overlay = cv2.addWeighted(frame, 1, cv2.cvtColor(occ, cv2.COLOR_GRAY2BGR), 0.5, 0)
        total = np.sum(occupancy_grid_display)
        curr_time = time.time()
        if total == 0:
            if curr_time - time_of_frame < buffer_time:
                occupancy_grid_display = memory_buffer
            else:
                occupancy_grid_display.fill(255)
        else:
            memory_buffer = occupancy_grid_display

        #cv2.imshow('Original Video', overlay)
            
        

        transformed_image, bottomLeft, bottomRight, topRight, topLeft, maxWidth, maxHeight  = getBirdView(occupancy_grid_display, ZED)

        #Transform video to bird's eye view
        transform_vid = getBirdView(frame, ZED)
        #cv2.imshow('Transformed Video', transform_vid[0])

        maxHeight = int(maxHeight)
        maxWidth = int(maxWidth)

        mask = np.full((maxHeight, maxWidth), -1, dtype=np.int8)
        # pts =  np.array([bottomLeft, [bottomRight[0] - 27, bottomRight[1]], [topRight[0] - 65, topRight[1]], topLeft])
        pts =  np.array([bottomLeft, [bottomRight[0] - 17, bottomRight[1]], [topRight[0] - 75, topRight[1]], topLeft])
        pts = pts.astype(np.int32)  # convert points to int32
        pts = pts.reshape((-1, 1, 2))  # reshape points
        cv2.fillPoly(mask, [pts], True, 0)

        indicies = np.where(mask == -1)
        transformed_image[indicies] = -1

        add_neg = np.full((transformed_image.shape[0], 66), -1, dtype=np.int8)

        transformed_image = np.concatenate((add_neg, transformed_image), axis=1)

        #(transformed_image.shape)
        

        transformed_image = np.where(transformed_image==255, 1, transformed_image)
        transformed_image = np.where((transformed_image != 0) & (transformed_image != 1) & (transformed_image != -1), -1, transformed_image)
        #print(bottomLeft, bottomRight)


        # np.savetxt('mask.txt', mask, fmt='%d')
        # np.savetxt('transformed_image.txt', transformed_image, fmt='%d')

        transformed_color = apply_custom_color_map(transformed_image)
        #cv2.imshow('Occupancy', transformed_color)

        
        current_pixel_size = 0.006  # current size each pixel represents in meters
        desired_pixel_size = 0.05  # desired size each pixel should represent in meters
        scale_factor = current_pixel_size / desired_pixel_size

        # Resize the transformed image
        new_size = (int(transformed_image.shape[1] * scale_factor), int(transformed_image.shape[0] * scale_factor))
        resized_image = cv2.resize(transformed_image, new_size, interpolation = cv2.INTER_NEAREST_EXACT)
        #(new_size)

        # np.savetxt('resized_image.txt', resized_image, fmt='%d')


        rob_arr = np.full((22, 169), -1, dtype=np.int8)
        rob_arr[10][85] = 2


        # combined_arr = np.concatenate((resized_image, rob_arr), axis=0)
        combined_arr = np.vstack((resized_image, rob_arr))

        # combined_arr = combined_arr.astype(np.int8)
        #np.savetxt('combined_arr.txt', combined_arr, fmt='%d')

        unique_values = np.unique(combined_arr)
        #print(unique_values)

        combined_arr_color = apply_custom_color_map(combined_arr)



        # cv2.imshow('Occupancy Grid', combined_arr_color)

        combined_arr = np.where(combined_arr==0, 3, combined_arr)
        combined_arr = np.where(combined_arr==1, 0, combined_arr)
        combined_arr = np.where(combined_arr==3, 1, combined_arr)

        # np.savetxt('combined_arr.txt', combined_arr, fmt='%d')

        #show all the windows in one screen

        overlay = cv2.resize(overlay, (transformed_image.shape[1], 600))
        transform_vid = cv2.resize(transform_vid[0], (transformed_image.shape[1], transformed_image.shape[0]))
        transformed_color = cv2.resize(transformed_color, (transformed_image.shape[1], transformed_image.shape[0]))

        combined_all = np.concatenate((overlay, transform_vid, transformed_color), axis=0)
        cv2.imshow('All', combined_all)
        
        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('comp23_segmented.mp4', fourcc, 30.0, (combined_all.shape[1], combined_all.shape[0]))
        out.write(combined_all)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()