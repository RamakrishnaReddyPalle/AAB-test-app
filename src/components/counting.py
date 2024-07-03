import numpy as np

def count_boxes_by_class(filtered_classes):
    unique_classes, counts = np.unique(filtered_classes, return_counts=True)
    return dict(zip(unique_classes, counts))
