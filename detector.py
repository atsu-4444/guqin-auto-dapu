import cv2
from ultralytics import YOLO

def _sort_boxes_horizontal_reading(boxes, row_tolerance):
    """
    Sort bounding boxes for horizontal text layout (left-to-right, top-to-bottom).
    """
    if not boxes:
        return []
    
    boxes.sort(key=lambda box: box[1])
    rows, current_row = [], [boxes[0]]
    
    for box in boxes[1:]:
        if abs(box[1] - current_row[-1][1]) < row_tolerance:
            current_row.append(box)
        else:
            current_row.sort(key=lambda b: b[0])
            rows.append(current_row)
            current_row = [box]
            
    current_row.sort(key=lambda b: b[0])
    rows.append(current_row)

    return [box for row in rows for box in row]

def _sort_boxes_vertical_reading(boxes, column_tolerance):
    """
    Sort bounding boxes for vertical text layout (top-to-bottom, right-to-left).
    """
    if not boxes:
        return []

    boxes.sort(key=lambda box: box[0])
    columns, current_column = [], [boxes[0]]
    
    for box in boxes[1:]:
        if abs(box[0] - current_column[-1][0]) < column_tolerance:
            current_column.append(box)
        else:
            current_column.sort(key=lambda b: b[1])
            columns.append(current_column)
            current_column = [box]
            
    current_column.sort(key=lambda b: b[1])
    columns.append(current_column)

    # Sort columns from right to left
    columns.sort(key=lambda col: sum(b[0] for b in col) / len(col), reverse=True)

    return [box for col in columns for box in col]


class YOLODetector:
    def __init__(self, model_path):
        """
        Initialize YOLOv11 model.
        """
        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)

    def detect_and_sort(self, image_path, orientation, conf=0.3):
        """
        Detect Guqin characters and sort them based on reading orientation.
        """
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Error: Could not read image at {image_path}")
            return [], None

        results = self.model.predict(source=original_image, verbose=False, conf=conf)
        boxes_list = [box.tolist() for box in results[0].boxes.xyxy]
        img_height, img_width, _ = original_image.shape

        if orientation == 'horizontal':
            dynamic_tolerance = img_height * 0.05
            sorted_boxes = _sort_boxes_horizontal_reading(boxes_list, dynamic_tolerance)
        elif orientation == 'vertical':
            dynamic_tolerance = img_width * 0.03
            sorted_boxes = _sort_boxes_vertical_reading(boxes_list, dynamic_tolerance)
        else:
            print(f"Error: Unknown orientation mode '{orientation}'")
            return [], original_image

        return sorted_boxes, original_image