import cv2

def draw_boxes(image, boxes, scores, classes, conf_threshold=0.6):
    for box, score, cls in zip(boxes, scores, classes):
        if score >= conf_threshold:
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if cls == 0 else (255, 0, 0)
            label = f"{cls} {score:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image
