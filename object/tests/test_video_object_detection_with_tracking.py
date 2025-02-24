import pytest
import cv2
import numpy as np
import os
import pandas as pd
from object.video_object_detection_with_tracking import (
    load_yolo_model,
    process_video_frames_with_tracking,
    save_detection_video_with_tracking,
    save_tracking_results_to_csv,
    TrackedObject
)

@pytest.fixture
def yolo_model():
    """Fixture to provide loaded YOLO model."""
    return load_yolo_model()

@pytest.fixture
def test_video_path():
    """Fixture for test video path."""
    return "./object/tests/fixtures/videos/sample_2.mp4"

@pytest.fixture
def output_video_path(tmp_path):
    """Fixture for temporary output video path."""
    return str(tmp_path / "output_test.mp4")

@pytest.fixture
def output_csv_path(tmp_path):
    """Fixture for temporary CSV output path."""
    return str(tmp_path / "tracking_results.csv")

def test_model_loading():
    """Test YOLO model loads correctly."""
    model = load_yolo_model()
    assert model is not None
    assert hasattr(model, 'predict')

def test_tracked_object_creation():
    """Test TrackedObject dataclass creation and initialization."""
    tracked_obj = TrackedObject(
        track_id=1,
        class_id=0,
        confidence=0.95,
        bbox={"x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 200.0},
        center_point=(50, 100)
    )
    
    assert tracked_obj.track_id == 1
    assert tracked_obj.class_id == 0
    assert tracked_obj.confidence == 0.95
    assert tracked_obj.trace is not None
    assert len(tracked_obj.trace) == 1
    assert tracked_obj.trace[0] == (50, 100)

def test_process_video_frames_with_tracking(yolo_model, test_video_path):
    """Test video processing with tracking."""
    frame_generator = process_video_frames_with_tracking(
        test_video_path,
        yolo_model,
        conf_threshold=0.5
    )
    
    # Test first frame results
    first_result = next(frame_generator)
    
    assert "frame" in first_result
    assert "tracked_objects" in first_result
    assert "frame_number" in first_result
    
    assert isinstance(first_result["frame"], np.ndarray)
    assert isinstance(first_result["tracked_objects"], list)
    assert isinstance(first_result["frame_number"], int)
    
    if first_result["tracked_objects"]:
        obj = first_result["tracked_objects"][0]
        assert isinstance(obj, TrackedObject)
        assert obj.class_id == 0  # Person class
        assert 0 <= obj.confidence <= 1
        assert len(obj.trace) > 0

def test_person_only_detection(yolo_model, test_video_path):
    """Test that only people are detected and tracked."""
    for result in process_video_frames_with_tracking(test_video_path, yolo_model):
        for obj in result["tracked_objects"]:
            assert obj.class_id == 0, "Non-person object detected"

def test_tracking_consistency(yolo_model, test_video_path):
    """Test that tracking IDs remain consistent."""
    track_ids = set()
    last_frame_objects = {}
    
    for result in process_video_frames_with_tracking(test_video_path, yolo_model):
        current_objects = {obj.track_id: obj for obj in result["tracked_objects"]}
        
        # Check that existing tracks maintain reasonable positions
        for track_id, current_obj in current_objects.items():
            if track_id in last_frame_objects:
                last_obj = last_frame_objects[track_id]
                # Check position hasn't changed too drastically
                last_center = last_obj.center_point
                current_center = current_obj.center_point
                distance = np.sqrt(
                    (last_center[0] - current_center[0])**2 +
                    (last_center[1] - current_center[1])**2
                )
                assert distance < 100, "Tracking jump detected"
        
        track_ids.update(current_objects.keys())
        last_frame_objects = current_objects
        
        if len(track_ids) >= 5:  # Break after finding sufficient tracks
            break
    
    assert len(track_ids) > 0, "No tracks found"

def test_csv_export(yolo_model, test_video_path, output_csv_path):
    """Test CSV export functionality."""
    # Generate tracking results
    results = []
    for i, result in enumerate(process_video_frames_with_tracking(test_video_path, yolo_model)):
        results.append(result)
        if i >= 10:  # Test with first 10 frames
            break
    
    # Save to CSV
    save_tracking_results_to_csv(results, output_csv_path)
    
    # Verify CSV contents
    assert os.path.exists(output_csv_path)
    df = pd.read_csv(output_csv_path)
    
    # Check structure
    required_columns = [
        'frame_number', 'track_id', 'class_id', 'confidence',
        'x1', 'y1', 'x2', 'y2', 'center_x', 'center_y'
    ]
    assert all(col in df.columns for col in required_columns)
    
    # Check data validity
    assert len(df) > 0
    assert df['class_id'].unique() == [0]  # Only person class
    assert (df['confidence'] >= 0).all() and (df['confidence'] <= 1).all()
    assert (df['x2'] > df['x1']).all()
    assert (df['y2'] > df['y1']).all()

def test_video_saving(yolo_model, test_video_path, output_video_path):
    """Test video saving with tracking visualization."""
    save_detection_video_with_tracking(
        video_path=test_video_path,
        output_path=output_video_path,
        model=yolo_model,
        conf_threshold=0.5,
        skip_frames=0,
        show_preview=False,
        save_csv=True
    )
    
    assert os.path.exists(output_video_path)
    csv_path = output_video_path.replace('.mp4', '_tracking.csv')
    assert os.path.exists(csv_path)
    
    # Verify video is readable
    cap = cv2.VideoCapture(output_video_path)
    assert cap.isOpened()
    ret, frame = cap.read()
    assert ret
    assert frame is not None
    cap.release()
