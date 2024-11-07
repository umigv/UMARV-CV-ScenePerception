import numpy as np

mask_matrix = np.array(
    [
        [1, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1]
    ])

def endpoints(mask_matrix):
    # First, we scan the datapoint to determine the lane line position in the edge
    left_lane_pos = right_lane_pos = bottommost_pos_range = topmost_pos_range = None
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
    print(edge_ones)
    # We can examine the edge_ones dict to see how lane lines are located
    if edge_ones.get('top'):
        # If there are lane lines at the top
        # 1. Put the leftmost of the lane line to the position
        left_lane_pos

def main(mask_matrix):
    return endpoints(mask_matrix)

if __name__ == "__main__":
    main(mask_matrix)