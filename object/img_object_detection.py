from typing import List, Dict, Union
from ultralytics import YOLO

def load_yolo_model() -> YOLO:
    model = YOLO("./object/tests/fixtures/models/yolo11l.pt")
    return model

def detect_objects(image_path: str, model: YOLO, conf_threshold: float = 0.5) -> List[Dict[str, Union[int, float]]]:
    """
    Detect objects in an image using the YOLO model.

    Args:
        image_path (str): Path to the input image.
        model (YOLO): Preloaded YOLO model.
        conf_threshold (float): Confidence threshold for filtering detections.

    Returns:
        List[Dict[str, Union[int, float]]]: List of detected objects with class IDs, confidence scores, and bounding boxes.
    """
    results = model.predict(image_path, conf=conf_threshold)
    
    detections = []
    for box in results[0].boxes.data.tolist():  # Extract detection data
        x1, y1, x2, y2, confidence, class_id = box
        detections.append({
            "class_id": int(class_id),
            "confidence": float(confidence),
            "bbox": {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2)
            }
        })

    return detections


results = detect_objects('./object/tests/fixtures/images/friends.jpg', load_yolo_model())
print(results)
