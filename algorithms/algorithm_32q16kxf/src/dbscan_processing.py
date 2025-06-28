from sklearn.cluster import DBSCAN
from utils import convertTorch2DToDBSCANInput, convertListOfPointsToMask

# mask should be a binary 2d array
def apply_DBSCAN(mask, eps, min_samples):
    positive_idx, negative_idx = convertTorch2DToDBSCANInput(mask)
    positive_labels = DBSCAN(eps=eps, min_samples=min_samples).fit(positive_idx).labels_
    negative_labels = DBSCAN(eps=eps, min_samples=min_samples).fit(negative_idx).labels_
    new_mask = convertListOfPointsToMask(positive_labels, negative_labels, positive_idx, negative_idx, mask.shape)
    return new_mask
    