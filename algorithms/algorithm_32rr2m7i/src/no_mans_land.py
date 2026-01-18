from hsv import hsv
import numpy as np
import cv2

class NoMansLand(hsv):
    def __init__(self):
        super().__init__()

    def find_line_cols(self, row, occupancy_grid):
            line1 = 0
            in_a_row = 0
            for col in range(occupancy_grid.shape[1]):
                if (occupancy_grid[row, col] > 0):
                    in_a_row += 1
                    if(in_a_row == 5):
                        line1 = col
            line2 = self.image_width-10
            in_a_row = 0
            for col in range(line1 + 200, occupancy_grid.shape[1]):
                if (occupancy_grid[row, col] > 0):
                    in_a_row += 1
                    if(in_a_row == 5):
                        line2 = col
            return line1, line2
                    
        
    def find_closest_row(self, occupancy_grid):
            # Iterate from the bottom to the top
        in_a_row = 0
        for row in range(occupancy_grid.shape[0] - 1, -1, -1):
            if np.any(occupancy_grid[row, :] > 0):
                in_a_row += 1
                if(in_a_row == 7):
                    return row-50
            else:
                in_a_row = 0
        return None
        
    def nomansland_func(self, combined, mask_dict):
        occupancy_grid = mask_dict['white']
        closest_row = self.find_closest_row(occupancy_grid)
        if closest_row is not None:
            l1, l2 = self.find_line_cols(closest_row, occupancy_grid=occupancy_grid)
            cv2.line(occupancy_grid, (l1, closest_row), (0, occupancy_grid.shape[0]), (255, 0, 0), 10)  # Line to bottom-left corner
            cv2.line(occupancy_grid, (l2, closest_row), (occupancy_grid.shape[1], occupancy_grid.shape[0]), (255, 0, 0), 10)  # Line to bottom-right corner
            cv2.line(combined, (l1, closest_row), (0, occupancy_grid.shape[0]), (255, 0, 0), 10)  # Line to bottom-left corner
            cv2.line(combined, (l2, closest_row), (occupancy_grid.shape[1], occupancy_grid.shape[0]), (255, 0, 0), 10)  # Line to bottom-right corner
        

    def get_mask(self, frame, yolo=False, nomandsland=False):
        self.image = frame
        self.adjust_gamma()
        self.hsv_image = cv2.cvtColor(self.gamma_image, cv2.COLOR_BGR2HSV)

        if nomandsland:
            combined, mask_dict = self.update_mask()
            self.nomansland_func(combined, mask_dict)
            return combined, mask_dict
        
        if yolo:
            self.get_lane_lines_YOLO()
            mask, mask_dict = self.update_mask()
            mask = cv2.bitwise_or(mask, self.laneline_mask)
            mask_dict['yolo'] = self.laneline_mask
            if(mask_dict.find('white') != -1):
                mask_dict['white'] = cv2.bitwise_or(mask_dict['white'], self.laneline_mask)
            return mask, mask_dict
        else:
            return self.update_mask()

if __name__ == "__main__":
    hsv_obj = hsv("data/IMG_5102.MOV")
    hsv_obj.tune("white")

