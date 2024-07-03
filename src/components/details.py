def get_predicted_boxes_with_details(filtered_boxes, filtered_scores, filtered_classes):
    detailed_boxes = []
    for box, score, cls in zip(filtered_boxes, filtered_scores, filtered_classes):
        detailed_boxes.append({
            "box": box.tolist(),
            "score": score,
            "class": int(cls)
        })
    return detailed_boxes
