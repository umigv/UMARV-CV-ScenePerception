import cv2 as cv
import numpy as np
from scipy.spatial import ConvexHull
import supervision as sv
from shapely import MultiPoint, concave_hull, STRtree

# Define functions to convert data into the correct formats for DBSCAN/OpenCV binary masks
def convertTorch2DToDBSCANInput(x):
    ones_idx = np.transpose((x > 0).nonzero())
    zeros_idx = np.transpose((x == 0).nonzero())
    return ones_idx, zeros_idx
     
# Returns a dictionary of labels and convex hulls
def getConvexHullsOfClusters(idx, labels):
    convex_hulls = {}
    for cluster_id in set(labels):
        if cluster_id > -1:
            cluster_idx = idx[labels==cluster_id]
            hull = ConvexHull(cluster_idx)
            convex_hulls[cluster_id] = cluster_idx[hull.vertices]
    return convex_hulls

# Same as above, except it doesn't remove concave portions of hull
def getBoundarySetsOfClusters(idx, labels):
    boundary_sets = {}
    for cluster_id in set(labels):
        if cluster_id > -1:
            cluster_idx = idx[labels==cluster_id]
            boundary_points = concave_hull(MultiPoint(cluster_idx), ratio=0.1)
            #boundary_points = concave_hull(cluster_idx, concavity=4.0)#, length_threshold=50) # TODO: Mess with these parameters
            boundary_sets[cluster_id] = boundary_points
    return boundary_sets

def applyBoundarySetsToMask(positive_boundary_sets, negative_boundary_sets, shape):
    grid = np.zeros(shape, dtype=np.uint8)
    if len(positive_boundary_sets) == 0:
        # Base case, return empty grid
        return grid
    if len(negative_boundary_sets) == 0:
        # Simple case, just apply all positive boundary sets in no particular order
        for cluster_id, boundary_set in positive_boundary_sets.items():
            grid = np.logical_or(grid, sv.polygon_to_mask(np.array(boundary_set.exterior.coords, dtype=np.int32),shape).astype(np.uint8).T).astype(np.uint8)
        return grid
    # Otherwise, there is at least one negative boundary set and at least one positive boundary set
    # TODO: This part is not done yet

    """
        Create tree where children are entirely contained by the parent
        1) Add a positive boundary set as the root
        2) Iterate through all positive boundary sets. If a
    """
    positive_tree = STRtree(positive_boundary_sets) # Base it off of this, way too slow otherwise
    negative_tree = STRtree(negative_boundary_sets) # Base it off of this, way too slow otherwise

    # Add all positive boundary sets that aren't contained by other positive sets
    added = set()
    for cluster_id, boundary_set in positive_boundary_sets:
        contained = False
        for cluster_id2, boundary_set2 in positive_boundary_sets:
            if cluster_id != cluster_id2:
                if boundary_set2.contains(boundary_set):
                    contained = True
                    break
        if not contained:
            # Clear to add boundary set to grid
            added.add(boundary_set)
            grid = grid = np.logical_or(grid, sv.polygon_to_mask(boundary_set.exterior.coords.xy,shape).T)

    # Next, remove all negative boundary sets that are contained by one of the added positive sets but are not contained by another negative set
    for cluster_id, boundary_set in negative_boundary_sets:
        contained = False
        for cluster_id2, boundary_set2 in added:
            if cluster_id != cluster_id2:
                if boundary_set2.contains(boundary_set):
                    contained = True
                    break
        if not contained:
            # Clear to add boundary set to grid
            added.add(boundary_set)
            grid = grid = np.logical_or(grid, sv.polygon_to_mask(boundary_set.exterior.coords.xy,shape).T)
    

def convertListOfPointsToMask(positive_labels, negative_labels, positive_idx, negative_idx, shape):
    positive_boundary_sets = getBoundarySetsOfClusters(positive_idx, positive_labels)
    negative_boundary_sets = []#getBoundarySetsOfClusters(negative_idx, negative_labels) # TODO: This isn't done yet
    #convex_hulls = getConvexHullsOfClusters(idx, labels)
    """
    grid = np.zeros(shape)
    for cluster_id, convex_hull in convex_hulls.items():
        grid = np.logical_or(grid, sv.polygon_to_mask(convex_hull,shape).T)
    return grid.astype(np.uint8)
    """
    return applyBoundarySetsToMask(positive_boundary_sets, negative_boundary_sets, shape)
    
def applyMask(img, mask):
    mask = mask * 255
    return cv.bitwise_and(img, img, mask=mask)

def erodeAndDilate(mask):
    erode_kernel = np.ones((3, 3), np.uint8) 
    dilate_kernel = np.ones((5, 5), np.uint8) 
    img_erosion = cv.erode(mask, erode_kernel, iterations=1)
    img_dilate = cv.dilate(img_erosion, dilate_kernel, iterations=2)
    img_erosion = cv.erode(mask, dilate_kernel, iterations=1)
    return img_dilate

def createDifferenceImg(mask1, mask2):
    return cv.cvtColor(cv.bitwise_not(cv.bitwise_xor(mask1*255, mask2*255)), cv.COLOR_GRAY2BGR)