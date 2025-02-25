import pytest
import cv2
import numpy as np
import os
import pandas as pd
from object.video_object_detection_with_tracking import (
    load_yolo_model,
    process_video_frames_with_tracking,
    save_detection_video_with_tracking,
    save_tracking_results_to_csv
)
from object.data_model import TrackedObject, MovementSpeed, BoundingBox, FrameResult

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
    bbox = BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=200.0)
    tracked_obj = TrackedObject(
        track_id=1,
        class_id=0,
        confidence=0.95,
        bbox=bbox,
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
    
    assert isinstance(first_result, FrameResult)
    assert hasattr(first_result, "frame")
    assert hasattr(first_result, "tracked_objects")
    assert hasattr(first_result, "frame_number")
    
    assert isinstance(first_result.frame, np.ndarray)
    assert isinstance(first_result.tracked_objects, list)
    assert isinstance(first_result.frame_number, int)
    
    if first_result.tracked_objects:
        obj = first_result.tracked_objects[0]
        assert isinstance(obj, TrackedObject)
        assert obj.class_id == 0  # Person class
        assert 0 <= obj.confidence <= 1
        assert len(obj.trace) > 0

def test_person_only_detection(yolo_model, test_video_path):
    """Test that only people are detected and tracked."""
    for result in process_video_frames_with_tracking(test_video_path, yolo_model):
        for obj in result.tracked_objects:
            assert obj.class_id == 0, "Non-person object detected"

def test_tracking_consistency(yolo_model, test_video_path):
    """Test that tracking IDs remain consistent."""
    track_ids = set()
    last_frame_objects = {}
    
    for result in process_video_frames_with_tracking(test_video_path, yolo_model):
        current_objects = {obj.track_id: obj for obj in result.tracked_objects}
        
        # Check that existing tracks maintain consistent positions
        for track_id, obj in current_objects.items():
            if track_id in last_frame_objects:
                last_obj = last_frame_objects[track_id]
                last_center = last_obj.center_point
                current_center = obj.center_point
                
                # Calculate distance between centers
                distance = np.sqrt(
                    (current_center[0] - last_center[0])**2 +
                    (current_center[1] - last_center[1])**2
                )
                
                # Movement between frames should be reasonable
                # (allowing for fast movement but not teleportation)
                assert distance < 100, f"Unreasonable movement for track {track_id}: {distance} pixels"
        
        # Update tracking info
        track_ids.update(current_objects.keys())
        last_frame_objects = current_objects
        
        # Break after processing enough frames
        if len(track_ids) >= 3 and len(result.tracked_objects) >= 3:
            break
    
    # Ensure we found some tracks
    assert len(track_ids) > 0, "No tracks found"

def test_csv_export(yolo_model, test_video_path, output_csv_path):
    """Test CSV export functionality."""
    # Process a few frames
    results = []
    for i, result in enumerate(process_video_frames_with_tracking(test_video_path, yolo_model)):
        results.append(result)
        if i >= 10:  # Process 10 frames
            break
    
    # Save to CSV
    save_tracking_results_to_csv(results, output_csv_path)
    
    # Verify CSV exists and has correct format
    assert os.path.exists(output_csv_path)
    
    # Read CSV and check structure
    df = pd.read_csv(output_csv_path)
    
    # Check required columns
    required_columns = [
        'frame_number', 'timestamp', 'track_id', 'class_id', 'confidence',
        'x1', 'y1', 'x2', 'y2', 'center_x', 'center_y',
        'velocity_x', 'velocity_y', 'speed', 'direction', 'movement_type'
    ]
    
    for col in required_columns:
        assert col in df.columns, f"Missing column: {col}"
    
    # Check data types and values
    assert df['frame_number'].dtype in [np.int64, int]
    assert df['track_id'].dtype in [np.int64, int]
    assert df['class_id'].dtype in [np.int64, int]
    assert all(df['class_id'] == 0)  # All should be person class
    assert (0 <= df['confidence'].astype(float)).all() and (df['confidence'].astype(float) <= 1).all()
    assert df['movement_type'].isin([e.value for e in MovementSpeed]).all()

def test_movement_classification(yolo_model, test_video_path):
    """Test movement classification thresholds."""
    frame_generator = process_video_frames_with_tracking(
        test_video_path,
        yolo_model,
        conf_threshold=0.5
    )
    
    movement_types_found = set()
    
    for result in process_video_frames_with_tracking(test_video_path, yolo_model):
        for obj in result.tracked_objects:
            if obj.movement_type:
                movement_types_found.add(obj.movement_type)
                
                # Verify speed matches movement type
                if obj.movement_type == MovementSpeed.STATIONARY:
                    assert obj.speed < 50
                elif obj.movement_type == MovementSpeed.WALKING:
                    assert 50 <= obj.speed < 200
                elif obj.movement_type == MovementSpeed.RUNNING:
                    assert obj.speed >= 200
                    
        if len(movement_types_found) >= 2:  # Found at least two different movement types
            break
    
    assert len(movement_types_found) > 0, "No movement classifications found"

def test_velocity_consistency(yolo_model, test_video_path):
    """Test that velocity calculations are consistent over time."""
    last_velocities = {}
    
    for result in process_video_frames_with_tracking(test_video_path, yolo_model):
        for obj in result.tracked_objects:
            if obj.track_id in last_velocities and obj.velocity:
                last_vel = last_velocities[obj.track_id]
                current_vel = obj.velocity
                
                # Velocity shouldn't change too drastically between frames
                if last_vel and current_vel:
                    vel_change = np.sqrt(
                        (current_vel[0] - last_vel[0])**2 +
                        (current_vel[1] - last_vel[1])**2
                    )
                    # Allow for more realistic velocity changes (roughly 8-9 m/s in pixel space)
                    assert vel_change < 1000, f"Unrealistic velocity change detected: {vel_change:.2f} px/s"
                    
                    # Additional sanity checks
                    assert not np.isnan(vel_change), "NaN velocity detected"
                    assert np.isfinite(vel_change), "Infinite velocity detected"
            
            if obj.velocity:
                last_velocities[obj.track_id] = obj.velocity

def test_tracked_object_initialization():
    """Test TrackedObject initialization with movement metrics."""
    bbox = BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=200.0)
    obj = TrackedObject(
        track_id=1,
        class_id=0,
        confidence=0.95,
        bbox=bbox,
        center_point=(50, 100)
    )
    
    assert obj.speed is None
    assert obj.direction is None
    assert obj.movement_type is None
    assert obj.velocity is None
    assert obj.trace is not None
    assert len(obj.trace) == 1

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
