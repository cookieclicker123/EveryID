from typing import List
import sys
import os

# Add the project root to the path when running as a script
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultralytics import YOLO
from object.data_model import Detection, ModelConfig, BoundingBox

def load_yolo_model() -> YOLO:
    """Load YOLO model from fixtures."""
    model_config = ModelConfig(model_path="./object/tests/fixtures/models/yolo11l.pt")
    return model_config.load_model()

def detect_objects(image_path: str, model: YOLO, conf_threshold: float = 0.5) -> List[Detection]:
    """
    Detect objects in an image using the YOLO model.

    Args:
        image_path (str): Path to the input image.
        model (YOLO): Preloaded YOLO model.
        conf_threshold (float): Confidence threshold for filtering detections.

    Returns:
        List[Detection]: List of detected objects with class IDs, confidence scores, and bounding boxes.
    """
    results = model.predict(image_path, conf=conf_threshold)
    
    detections = []
    for box in results[0].boxes.data.tolist():  # Extract detection data
        x1, y1, x2, y2, confidence, class_id = box
        detections.append(Detection(
            class_id=int(class_id),
            confidence=float(confidence),
            bbox=BoundingBox(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2)
            )
        ))

    return detections


if __name__ == "__main__":
    results = detect_objects('./object/tests/fixtures/images/friends.jpg', load_yolo_model())
    print(results)
