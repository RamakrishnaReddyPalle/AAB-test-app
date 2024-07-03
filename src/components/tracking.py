import numpy as np
from src.utils import compute_iou, merge_boxes

def track_and_filter_predictions(predictions, tile_coords, iou_threshold=0.5):
    tracked_objects = {}

    for pred, (tile_x, tile_y) in zip(predictions, tile_coords):
        if len(pred[0]) > 0:
            boxes = pred[0].boxes.xyxy.cpu().numpy()
            scores = pred[0].boxes.conf.cpu().numpy()
            classes = pred[0].boxes.cls.cpu().numpy()

            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                adjusted_box = [x1 + tile_x, y1 + tile_y, x2 + tile_x, y2 + tile_y]
                key = (int(cls), tuple(map(int, adjusted_box)))

                updated = False
                for tracked_key in list(tracked_objects.keys()):
                    tracked_cls, tracked_box = tracked_key
                    if tracked_cls == cls and compute_iou(adjusted_box, tracked_box) > iou_threshold:
                        if tracked_objects[tracked_key] < score:
                            tracked_objects[tracked_key] = score
                            new_merged_box = merge_boxes(adjusted_box, tracked_box)
                            tracked_key_new = (tracked_cls, tuple(new_merged_box))
                            print(f"Merging {tracked_box} with {adjusted_box} into {new_merged_box}")
                            tracked_objects[tracked_key_new] = tracked_objects.pop(tracked_key)
                        updated = True
                        break

                if not updated:
                    tracked_objects[key] = score

    filtered_boxes = []
    filtered_scores = []
    filtered_classes = []

    for (cls, box), score in tracked_objects.items():
        filtered_boxes.append(box)
        filtered_scores.append(score)
        filtered_classes.append(cls)

    # Further filter based on three-coordinate similarity
    filtered_boxes, filtered_scores, filtered_classes = filter_by_coordinate_similarity(filtered_boxes, filtered_scores, filtered_classes)

    return np.array(filtered_boxes), np.array(filtered_scores), np.array(filtered_classes)

def filter_by_coordinate_similarity(boxes, scores, classes, coord_threshold=15, diff_threshold=30):
    filtered_boxes = []
    filtered_scores = []
    filtered_classes = []

    def coordinates_similar(box1, box2, threshold=coord_threshold):
        x1_similar = abs(box1[0] - box2[0]) <= threshold
        y1_similar = abs(box1[1] - box2[1]) <= threshold
        x2_similar = abs(box1[2] - box2[2]) <= threshold
        y2_similar = abs(box1[3] - box2[3]) <= threshold
        return x1_similar and y1_similar and x2_similar and y2_similar

    def dimensions_diff(box1, box2, threshold=diff_threshold):
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        w_diff = abs(w1 - w2)
        h_diff = abs(h1 - h2)
        return w_diff <= threshold and h_diff <= threshold

    for i, box1 in enumerate(boxes):
        is_duplicate = False
        for j, box2 in enumerate(filtered_boxes):
            if coordinates_similar(box1, box2) and dimensions_diff(box1, box2):
                is_duplicate = True
                break

        if not is_duplicate:
            filtered_boxes.append(box1)
            filtered_scores.append(scores[i])
            filtered_classes.append(classes[i])

    return filtered_boxes, filtered_scores, filtered_classes
