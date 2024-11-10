"""
Take in a numpy 2D array as a mask, and then output the turning angle

Daniel X He <xinzhouh@umich.edu>

"""
import numpy as np

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

def split_by_discontinuity(lst, threshold):
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
            if edge_ones.get('bot') is not None:
                edge_ones['bot'].append(int(positions[row][1]))
            else:
                edge_ones['bot'] = []
                edge_ones['bot'].append(int(positions[row][1]))
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
            edge_ones[key] = split_by_discontinuity(list_edge_one, 1)
        # end if
    # end for

    # Return the dictionary containing the endpoints info
    return edge_ones
# end def

def compute_angle(endpoints):
    # W
    # First, we check if each edge has the endpoints
    if endpoints.get(endpoints):
        # If the top edge has the endpoints
        # Check if it is divided
        if len(endpoints[endpoints]) > 1:
            # If so, we assume they are the endpoints of the left/right lanes
            ...
    ...

def main(mask_matrix):
    # First, make a copy of the mask to avoid the change of the original structure
    mask_matrix_copy = mask_matrix
    # Get the endpoints of the mask
    endpoints_info = endpoints(mask_matrix_copy)
    # Calculate the angle position
    angle_positions = compute_angle(endpoints_info)

    # Return the final results
    return angle_positions

if __name__ == "__main__":
    main(mask_matrix)