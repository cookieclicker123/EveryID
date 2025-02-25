from typing import List, Dict, Generator, Optional, Tuple
import cv2
import numpy as np
import sys
import os

# Add the project root to the path when running as a script
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultralytics import YOLO
import supervision as sv
import csv
import math
from object.data_model import (
    TrackedObject, MovementSpeed, BoundingBox, 
    FrameResult, ModelConfig, VideoProcessor
)

# Constants for visualization
TRACE_LENGTH = 30
TRACE_THICKNESS = 2
TRACE_FADE_FACTOR = 0.95

def load_yolo_model() -> YOLO:
    """Load YOLO model from fixtures."""
    model_config = ModelConfig(model_path="./object/tests/fixtures/models/yolo11l.pt")
    return model_config.load_model()

def calculate_center(bbox: BoundingBox) -> Tuple[int, int]:
    """Calculate center point of bounding box."""
    return (int(bbox.center[0]), int(bbox.center[1]))

class PersonTrackingProcessor(VideoProcessor):
    """Video processor for person tracking."""
    
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.tracker = sv.ByteTrack()
        self.tracked_objects = {}  # track_id -> TrackedObject
    
    def _process_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> FrameResult:
        """Process a single frame to detect and track people."""
        # Get detections
        results = self.model.predict(frame, conf=self.model_config.conf_threshold, verbose=False)[0]
        
        # Convert boxes, scores, and class_ids to numpy arrays
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        # Create Detections object directly
        detections = sv.Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=class_ids
        )
        
        # Filter for person class (class 0)
        mask = np.array([class_id == 0 for class_id in detections.class_id])
        detections = detections[mask]
        
        # Track detections
        tracked_detections = self.tracker.update_with_detections(detections)
        
        current_objects = []
        for det_idx in range(len(tracked_detections)):
            track_id = tracked_detections.tracker_id[det_idx]
            bbox = tracked_detections.xyxy[det_idx]
            confidence = tracked_detections.confidence[det_idx]
            
            # Convert bbox to our format
            bbox_obj = BoundingBox(
                x1=float(bbox[0]),
                y1=float(bbox[1]),
                x2=float(bbox[2]),
                y2=float(bbox[3])
            )
            
            center = calculate_center(bbox_obj)
            
            # Create or update tracked object
            if track_id in self.tracked_objects:
                tracked_obj = self.tracked_objects[track_id]
                tracked_obj.bbox = bbox_obj
                tracked_obj.confidence = confidence
                # Update velocity before updating center
                tracked_obj.update_velocity(center, timestamp)
                tracked_obj.center_point = center
                tracked_obj.trace.append(center)
            else:
                tracked_obj = TrackedObject(
                    track_id=track_id,
                    class_id=0,  # person
                    confidence=confidence,
                    bbox=bbox_obj,
                    center_point=center,
                    last_timestamp=timestamp
                )
                self.tracked_objects[track_id] = tracked_obj
            
            current_objects.append(tracked_obj)
        
        return FrameResult(
            frame=frame,
            tracked_objects=current_objects,
            frame_number=frame_number,
            timestamp=timestamp
        )

def process_video_frames_with_tracking(
    video_path: str,
    model: YOLO,
    conf_threshold: float = 0.5,
    skip_frames: int = 0
) -> Generator[FrameResult, None, None]:
    """Process video frames with object tracking."""
    if conf_threshold <= 0 or conf_threshold > 1:
        raise ValueError("Confidence threshold must be between 0 and 1")
    
    model_config = ModelConfig(
        model_path="",  # Not needed as model is already loaded
        conf_threshold=conf_threshold
    )
    processor = PersonTrackingProcessor(model_config)
    processor.model = model  # Use the provided model
    
    yield from processor.process_video(video_path, skip_frames)

def draw_traces(frame: np.ndarray, tracked_objects: List[TrackedObject]) -> np.ndarray:
    """Draw motion traces for tracked objects."""
    overlay = frame.copy()
    
    for obj in tracked_objects:
        trace = list(obj.trace)
        if len(trace) < 2:
            continue
        
        # Draw fading trace
        alpha = 1.0
        for i in range(len(trace) - 1):
            pt1 = trace[i]
            pt2 = trace[i + 1]
            
            color = (0, 255, 0)  # Green trace
            thickness = max(1, int(TRACE_THICKNESS * alpha))
            
            cv2.line(overlay, pt1, pt2, color, thickness)
            alpha *= TRACE_FADE_FACTOR
    
    # Blend overlay with original frame
    return cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

def save_tracking_results_to_csv(
    tracking_results: List[FrameResult],
    output_path: str = "./object/tests/fixtures/csv_object_tracking/tracking_results.csv"
) -> None:
    """Save tracking results to CSV format with enhanced movement metrics."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'frame_number',
            'timestamp',
            'track_id',
            'class_id',
            'confidence',
            'x1', 'y1', 'x2', 'y2',
            'center_x', 'center_y',
            'velocity_x', 'velocity_y',
            'speed',
            'direction',
            'movement_type'
        ])
        
        for result in tracking_results:
            frame_number = result.frame_number
            timestamp = result.timestamp
            for obj in result.tracked_objects:
                bbox = obj.bbox
                center_x, center_y = obj.center_point
                vel_x, vel_y = obj.velocity if obj.velocity else (0.0, 0.0)
                
                writer.writerow([
                    frame_number,
                    f"{timestamp:.3f}",
                    obj.track_id,
                    obj.class_id,
                    f"{obj.confidence:.3f}",
                    f"{bbox.x1:.2f}",
                    f"{bbox.y1:.2f}",
                    f"{bbox.x2:.2f}",
                    f"{bbox.y2:.2f}",
                    center_x,
                    center_y,
                    f"{vel_x:.2f}",
                    f"{vel_y:.2f}",
                    f"{obj.speed:.2f}" if obj.speed is not None else "0.00",
                    f"{obj.direction:.1f}" if obj.direction is not None else "0.0",
                    obj.movement_type.value if obj.movement_type else MovementSpeed.STATIONARY.value
                ])

def save_detection_video_with_tracking(
    video_path: str,
    output_path: str,
    model: YOLO,
    conf_threshold: float = 0.5,
    skip_frames: int = 0,
    show_preview: bool = False,
    save_csv: bool = True
) -> None:
    """Save video with tracking visualization and optionally save tracking data to CSV."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    
    tracking_results = []  # Store results for CSV export
    
    try:
        # Initialize annotator with trace
        box_annotator = sv.BoxAnnotator()
        trace_annotator = sv.TraceAnnotator(
            thickness=2,
            trace_length=20,
            position=sv.Position.CENTER
        )

        for result in process_video_frames_with_tracking(
            video_path, model, conf_threshold, skip_frames
        ):
            if save_csv:
                tracking_results.append(result)
                
            frame = result.frame
            tracked_objects = result.tracked_objects
            
            # Convert our tracked objects back to supervision format
            detections = sv.Detections(
                xyxy=np.array([obj.bbox.to_xyxy() for obj in tracked_objects]),
                confidence=np.array([obj.confidence for obj in tracked_objects]),
                class_id=np.array([obj.class_id for obj in tracked_objects]),
                tracker_id=np.array([obj.track_id for obj in tracked_objects])
            )
            
            # Draw both boxes and traces
            frame = box_annotator.annotate(frame, detections)
            frame = trace_annotator.annotate(frame, detections)
            
            out.write(frame)
            
            if show_preview:
                cv2.imshow('Tracking Preview', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
    finally:
        out.release()
        if show_preview:
            cv2.destroyAllWindows()
            
        # Save tracking results to CSV
        if save_csv and tracking_results:
            csv_path = output_path.replace('.mp4', '_tracking.csv')
            save_tracking_results_to_csv(tracking_results, csv_path)

if __name__ == "__main__":
    model = load_yolo_model()
    video_path = "./object/tests/fixtures/videos/sample_2.mp4"
    output_path = "./object/tests/fixtures/videos/sample_tracked.mp4"
    
    save_detection_video_with_tracking(
        video_path=video_path,
        output_path=output_path,
        model=model,
        conf_threshold=0.5,
        skip_frames=0,
        show_preview=True,
        save_csv=True
    ) 