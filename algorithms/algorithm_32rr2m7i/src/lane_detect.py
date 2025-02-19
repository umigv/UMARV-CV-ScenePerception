import cv2
import numpy as np

class LaneDetector:
    def __init__(self):
        self.image_path = 'curved_self_drive_left.png'
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError("Error: Could not load image. Check the file path.")

        self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.height, self.width = self.image.shape[:2]
        self.center_x = self.width // 2  # Vehicle's assumed center position

    def preprocess_image(self):
        """Preprocesses the image to highlight lane lines."""
        # Convert to HSV and isolate white color (adjusted range)
        lower_white = np.array([0, 0, 180])  
        upper_white = np.array([180, 50, 255])  

        mask = cv2.inRange(self.hsv_image, lower_white, upper_white)
        result = cv2.bitwise_and(self.image, self.image, mask=mask)

        # Convert to grayscale and apply Gaussian Blur
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        return edges, mask

    def detect_dashed_lines(self, edges):
        """Detects dashed lines in the image based on gaps."""
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=50, maxLineGap=200)
        dashed_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Analyze gaps in lines to detect dashed lines
                if abs(x2 - x1) > 50:  # Filter for horizontal lines (x1 ≈ x2)
                    dashed_lines.append((x1, y1, x2, y2))

        return dashed_lines

    def classify_lane(self, dashed_lines):
        """Classifies lanes based on dashed line position."""
        left_lane, right_lane = [], []
        if dashed_lines:
            for line in dashed_lines:
                x1, y1, x2, y2 = line
                # Calculate the position of the dashed line
                center_x = (x1 + x2) / 2
                # Classify lanes based on dashed line
                if center_x < self.center_x:  # Left of the center
                    left_lane.append(line)
                else:  # Right of the center
                    right_lane.append(line)

        return left_lane, right_lane

    def get_vehicle_position(self, left_lane, right_lane):
        """Determines the vehicle's position relative to the lanes."""
        if left_lane and right_lane:
            # Find the center of the lanes (average position of left and right lanes)
            left_x = np.mean([x1 for x1, _, x2, _ in left_lane] + [x2 for x1, _, x2, _ in left_lane])
            right_x = np.mean([x1 for x1, _, x2, _ in right_lane] + [x2 for x1, _, x2, _ in right_lane])
            lane_center = (left_x + right_x) / 2
            
            # Compare the vehicle's position with the lane center
            if self.center_x < lane_center:
                return "left"  # Vehicle is closer to the left lane
            elif self.center_x > lane_center:
                return "right"  # Vehicle is closer to the right lane
            else:
                return "center"  # Vehicle is in the center of the lanes
        elif left_lane:
            return "left"
        elif right_lane:
            return "right"
        else:
            return "unknown"  # No lanes detected

    def detect_lane_position(self):
        """Detects lane position and visualizes lane detection."""
        edges, mask = self.preprocess_image()

        # Detect dashed lines
        dashed_lines = self.detect_dashed_lines(edges)

        # Classify lanes based on dashed lines
        left_lane, right_lane = self.classify_lane(dashed_lines)

        # Get vehicle's position
        vehicle_position = self.get_vehicle_position(left_lane, right_lane)

        # Draw detected lanes for debugging
        output = self.image.copy()
        for x1, y1, x2, y2 in left_lane:
            cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue for left lane
        for x1, y1, x2, y2 in right_lane:
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green for right lane

        cv2.imshow("Lane Detection", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("Vehicle Position:", vehicle_position)

    def visualize_detection(self):
        """Draws detected lanes on the image."""
        output = self.image.copy()
        edges, mask = self.preprocess_image()

        dashed_lines = self.detect_dashed_lines(edges)
        left_lane, right_lane = self.classify_lane(dashed_lines)

        for x1, y1, x2, y2 in left_lane:
            cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for left lane
        for x1, y1, x2, y2 in right_lane:
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for right lane

        cv2.imshow("Lane Detection", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def main(self):
        self.detect_lane_position()
        self.visualize_detection()

if __name__ == '__main__':
    detector = LaneDetector()
    detector.main()
