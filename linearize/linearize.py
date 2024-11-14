"""
Take in a numpy 2D array as a mask, and then output the turning angle

Author(s): Daniel X He <xinzhouh@umich.edu>
Version: 1.0.0
Version Notes: This version works on the trivial case, where two lanes always starting from the bottom AND ending on the top,
      the next development aims to be able to figure out if the lane turned to the left/right.
Last Edit: November 14, 2024

"""
import numpy as np
import math

# Test data
mask_matrix = np.array(
    [
        [1, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1]
    ])

def split_by_discontinuity(lst, threshold=1):
    """Split the list if there is a discontinuity, helpful for later positional analysis."""
    if not lst:
        return []
    
    # Initialize the first sublist
    sublists = [[lst[0]]]
    
    for i in range(1, len(lst)):
        # Check if the difference between consecutive elements exceeds the threshold
        if abs(lst[i] - lst[i - 1]) > threshold:
            # Start a new sublist
            sublists.append([lst[i]])
        else:
            # Append to the current sublist
            sublists[-1].append(lst[i])
    
    return sublists

def endpoints(mask_matrix):
    """Scan and extract the endpoints info on the edge."""
    # Find all positions of '1's in the mask matrix
    positions = np.argwhere(mask_matrix == 1)
    # We use a dictionary to store positions
    edge_ones = {}
    for row in range(0, len(positions)):
        # Iterate for each row
        # If the row number is 0, the one is on the top edge
        if positions[row][0] == 0:
            if edge_ones.get('top') is not None:
                # We save the horizontal edge '1' position by their col index
                edge_ones['top'].append(int(positions[row][1]))
            else:
                edge_ones['top'] = []
                edge_ones['top'].append(int(positions[row][1]))
        # If the col number is 0, the one is on the left edge
        if positions[row][1] == 0:
            if edge_ones.get('left') is not None:
                # We save the vertical edge 1 position by their row index
                edge_ones['left'].append(int(positions[row][0]))
            else:
                edge_ones['left'] = []
                edge_ones['left'].append(int(positions[row][0]))
        # If the row number is len(mask_matrix) - 1, the one is on the bottom edge
        if positions[row][0] == len(mask_matrix) - 1:
            if edge_ones.get('bottom') is not None:
                edge_ones['bottom'].append(int(positions[row][1]))
            else:
                edge_ones['bottom'] = []
                edge_ones['bottom'].append(int(positions[row][1]))
        # If the row number is len(mask_matrix[0]) - 1, the one is on the right edge
        if positions[row][1] == len(mask_matrix[0]) - 1:
            if edge_ones.get('right') is not None:
                edge_ones['right'].append(int(positions[row][0]))
            else:
                edge_ones['right'] = []
                edge_ones['right'].append(int(positions[row][0]))
        # end if position
    # end for rows

    # Now edge_ones contain the 1D position of all 1s existed on each edge, if any
    # We can detect the discontinuity between the positions of 1s to infer the shape of the lane line
    for key, list_edge_one in edge_ones.items():
        if len(list_edge_one) > 1:
            # We do not need to 'split' any list if there is only one element
            edge_ones[key] = split_by_discontinuity(list_edge_one)
        # end if
    # end for

    # Return the dictionary containing the endpoints info
    return edge_ones
# end def

def compute_angle(mask_matrix, endpoints):
    # Make a copy of the data structure to avoid accidental modification
    mask_matrix_copy = mask_matrix
    endpoints_copy = endpoints
    # To store the lanes starting positions
    leftlane_start_pos = right_lane_start_pos = None
    # To store the lanes ending positions
    leftlane_end_pos = right_lane_end_pos = None

    # We loop through the dictionary to get the endpoints info
    for edge, clusters in endpoints_copy.items():
        # edge for each edge, clusters for the lane line info of the edge
        # We check if each edge has the endpoints
        if clusters:
            # If the edge has the endpoints
            # Check if it is divided, it is considered as divided if the number of clusters is more than 1
            if len(clusters) > 1:
                # If so, we assume they are the endpoints of the left/right lanes
                print(f'Two lanes ended on the {edge}')
                # If it is a bottom edge, we record the starting position
                if edge == 'bottom':
                    print(clusters)
                    # Pick the rightmost position of the left lane as its starting point
                    leftlane_start_pos = clusters[0][-1]
                    # And the leftmost position of the right lane as its starting point
                    right_lane_start_pos = clusters[1][0]
                elif edge == 'top':
                    # If the edge is the top, we record the ending point for lane
                    # Pick the leftmost position of the left lane as its starting point
                    leftlane_end_pos = clusters[0][0]
                    # And the rightmost position of the right lane as its starting point
                    right_lane_end_pos = clusters[1][-1]                    
                # end if edge side
            else:
                # We have exactly one lane line endpoints
                print(f'One lane line ends on the {edge}')
            # end if endpoints
        # end if edge has clusters
    # end for endpoints

    # Compute and output the angle, the 0 degree is set as the straight lane
    # angle increases when leaning right and decreases when leaning left
    # Equation: tan(theta) = Opposite (horizontal diff) / Adjacent (height)
    #           theta = tan^-1 (Opposite (horizontal diff) / Adjacent (height))
    opposite = leftlane_end_pos - leftlane_start_pos
    adjacent = len(mask_matrix_copy)
    print(f'Opposite: {opposite}\nAdjacent: {adjacent}')
    leftlane_angle = math.atan(opposite / adjacent)
    # Convert it to degrees
    leftlane_angle = math.degrees(leftlane_angle)
    print(f'Left lane angle: {leftlane_angle}')
    # Repeat the steps for the right lane
    opposite = right_lane_end_pos - right_lane_start_pos
    adjacent = len(mask_matrix_copy)
    print(f'Opposite: {opposite}\nAdjacent: {adjacent}')
    rightlane_angle = math.atan(opposite / adjacent)
    # Convert it to degrees
    rightlane_angle = math.degrees(rightlane_angle)
    print(f'Right lane angle: {rightlane_angle}')

    # Return the computed results as a pair
    return np.array([leftlane_angle, rightlane_angle], dtype=np.float64)
# end def

def main(mask_matrix):
    print(len(mask_matrix))
    # First, make a copy of the mask to avoid the change of the original structure
    mask_matrix_copy = mask_matrix
    # Get the endpoints of the mask
    endpoints_info = endpoints(mask_matrix_copy)
    # Calculate the angle position
    angle_positions = compute_angle(mask_matrix, endpoints_info)

    # Return the final results
    return angle_positions

if __name__ == "__main__":
    main(mask_matrix)