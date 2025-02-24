import pytest
from object.img_object_detection import load_yolo_model, detect_objects
import os

@pytest.fixture
def yolo_model():
    return load_yolo_model()

@pytest.fixture
def test_image_path():
    return os.path.join("object", "tests", "fixtures", "images", "friends.jpg")

def test_detect_objects(yolo_model, test_image_path):
    # Run detection
    results = detect_objects(test_image_path, yolo_model)
    
    # Basic validation
    assert isinstance(results, list)
    assert len(results) > 0
    
    # Validate structure of each detection
    for detection in results:
        assert isinstance(detection, dict)
        assert "class_id" in detection
        assert "confidence" in detection
        assert "bbox" in detection
        assert isinstance(detection["confidence"], float)
        assert 0 <= detection["confidence"] <= 1
        
        # Validate bbox structure
        bbox = detection["bbox"]
        assert all(key in bbox for key in ["x1", "y1", "x2", "y2"])
        assert all(isinstance(bbox[key], float) for key in bbox)

def test_detect_objects_expected_classes(yolo_model, test_image_path):
    results = detect_objects(test_image_path, yolo_model)
    
    # Known objects in the image
    person_detections = sum(1 for r in results if r["class_id"] == 0)  # person
    cup_detections = sum(1 for r in results if r["class_id"] == 41)    # cup
    
    # Test specific detections we know should be there
    assert person_detections >= 4, "Should detect at least 4 persons"
    assert cup_detections >= 3, "Should detect at least 3 cups"

def test_confidence_threshold(yolo_model, test_image_path):
    high_conf_threshold = 0.9
    results = detect_objects(test_image_path, yolo_model, conf_threshold=high_conf_threshold)
    
    # All detections should meet the confidence threshold
    for detection in results:
        assert detection["confidence"] >= high_conf_threshold

def test_invalid_image_path(yolo_model):
    with pytest.raises(Exception):
        detect_objects("nonexistent_image.jpg", yolo_model) 