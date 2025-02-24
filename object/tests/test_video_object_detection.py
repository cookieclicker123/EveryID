import pytest
import cv2
import numpy as np
import os
from object.video_object_detection import load_yolo_model, process_video_frames, save_detection_video
from ultralytics import YOLO

@pytest.fixture
def yolo_model():
    """Fixture to provide loaded YOLO model."""
    return load_yolo_model()

@pytest.fixture
def test_video_path():
    """Fixture for test video path."""
    return "./object/tests/fixtures/videos/sample.mp4"

@pytest.fixture
def output_video_path():
    """Fixture for output video path."""
    path = "./object/tests/fixtures/videos/test_output.mp4"
    yield path
    # Cleanup after test
    if os.path.exists(path):
        os.remove(path)

def test_model_loading():
    """Test YOLO model loads correctly."""
    model = load_yolo_model()
    assert isinstance(model, YOLO)
    assert model is not None

def test_video_capture_initialization(test_video_path):
    """Test video capture object initializes correctly."""
    cap = cv2.VideoCapture(test_video_path)
    assert cap.isOpened(), "Video file could not be opened"
    
    # Check video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    assert width > 0, "Invalid video width"
    assert height > 0, "Invalid video height"
    assert fps > 0, "Invalid video FPS"
    
    cap.release()

def test_frame_processing(yolo_model, test_video_path):
    """Test frame processing and detection format."""
    frame_generator = process_video_frames(
        test_video_path,
        yolo_model,
        conf_threshold=0.5
    )
    
    # Test first frame
    first_result = next(frame_generator)
    
    # Check result structure
    assert isinstance(first_result, dict)
    assert all(key in first_result for key in ["frame", "detections", "frame_number"])
    
    # Check frame properties
    assert isinstance(first_result["frame"], np.ndarray)
    assert len(first_result["frame"].shape) == 3  # Height, width, channels
    
    # Check detections format
    for detection in first_result["detections"]:
        assert isinstance(detection, dict)
        assert all(key in detection for key in ["class_id", "confidence", "bbox"])
        assert isinstance(detection["confidence"], float)
        assert 0 <= detection["confidence"] <= 1
        
        bbox = detection["bbox"]
        assert all(key in bbox for key in ["x1", "y1", "x2", "y2"])
        assert all(isinstance(bbox[key], float) for key in bbox)

def test_frame_skipping(yolo_model, test_video_path):
    """Test frame skipping functionality."""
    skip_frames = 2
    frame_numbers = []
    
    for result in process_video_frames(test_video_path, yolo_model, skip_frames=skip_frames):
        frame_numbers.append(result["frame_number"])
        if len(frame_numbers) >= 5:  # Check first 5 processed frames
            break
    
    # Verify frame numbers follow skipping pattern
    for i in range(1, len(frame_numbers)):
        assert frame_numbers[i] - frame_numbers[i-1] == skip_frames + 1

def test_video_saving(yolo_model, test_video_path, output_video_path):
    """Test video saving functionality."""
    save_detection_video(
        test_video_path,
        output_video_path,
        yolo_model,
        conf_threshold=0.5,
        skip_frames=0,
        show_preview=False
    )
    
    # Verify output file exists and is valid
    assert os.path.exists(output_video_path)
    cap = cv2.VideoCapture(output_video_path)
    assert cap.isOpened()
    
    # Check video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert width > 0 and height > 0
    
    # Check first frame
    ret, frame = cap.read()
    assert ret
    assert isinstance(frame, np.ndarray)
    assert frame.shape[2] == 3  # RGB channels
    
    cap.release()

def test_person_only_detection(yolo_model, test_video_path):
    """Test that only people (class 0) are detected."""
    frame_generator = process_video_frames(
        test_video_path,
        yolo_model,
        conf_threshold=0.5
    )
    
    # Test multiple frames
    person_detections_found = False
    for _ in range(10):  # Check first 10 frames
        try:
            result = next(frame_generator)
            
            # Verify all detections are people
            for detection in result["detections"]:
                assert detection["class_id"] == 0, "Non-person detection found"
                person_detections_found = True
                
        except StopIteration:
            break
    
    assert person_detections_found, "No person detections found in test video"

def test_non_person_filtering(yolo_model, test_video_path):
    """Test that non-person detections are filtered out."""
    frame_generator = process_video_frames(
        test_video_path,
        yolo_model,
        conf_threshold=0.3  # Lower threshold to ensure detections
    )
    
    # Get first frame with detections
    result = next(frame_generator)
    
    # Verify only person detections are present
    assert all(det["class_id"] == 0 for det in result["detections"]), \
        "Non-person detections found"
