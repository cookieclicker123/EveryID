from typing import List, Dict, Union, Generator, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import supervision as sv
from dataclasses import dataclass
import csv
import os

# Constants for visualization
TRACE_LENGTH = 30
TRACE_THICKNESS = 2
TRACE_FADE_FACTOR = 0.95

@dataclass
class TrackedObject:
    """Data class for tracked object information."""
    track_id: int
    class_id: int
    confidence: float
    bbox: Dict[str, float]
    center_point: Tuple[int, int]
    velocity: Optional[Tuple[float, float]] = None
    trace: deque = None
    
    def __post_init__(self):
        if self.trace is None:
            self.trace = deque(maxlen=TRACE_LENGTH)
        self.trace.append(self.center_point)

def load_yolo_model() -> YOLO:
    """Load YOLO model from fixtures."""
    model = YOLO("./object/tests/fixtures/models/yolo11l.pt")
    return model

def calculate_center(bbox: Dict[str, float]) -> Tuple[int, int]:
    """Calculate center point of bounding box."""
    x = int((bbox["x1"] + bbox["x2"]) / 2)
    y = int((bbox["y1"] + bbox["y2"]) / 2)
    return (x, y)

def process_video_frames_with_tracking(
    video_path: str,
    model: YOLO,
    conf_threshold: float = 0.5,
    skip_frames: int = 0
) -> Generator[Dict[str, Union[np.ndarray, List[TrackedObject]]], None, None]:
    """Process video frames with object tracking."""
    if conf_threshold <= 0 or conf_threshold > 1:
        raise ValueError("Confidence threshold must be between 0 and 1")
    
    # Initialize tracker
    tracker = sv.ByteTrack()
    tracked_objects = {}  # track_id -> TrackedObject
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if skip_frames and frame_count % (skip_frames + 1) != 0:
                continue
            
            # Get detections
            results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
            
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
            tracked_detections = tracker.update_with_detections(detections)
            
            current_objects = []
            for det_idx in range(len(tracked_detections)):
                track_id = tracked_detections.tracker_id[det_idx]
                bbox = tracked_detections.xyxy[det_idx]
                confidence = tracked_detections.confidence[det_idx]
                
                # Convert bbox to our format
                bbox_dict = {
                    "x1": float(bbox[0]),
                    "y1": float(bbox[1]),
                    "x2": float(bbox[2]),
                    "y2": float(bbox[3])
                }
                
                center = calculate_center(bbox_dict)
                
                # Create or update tracked object
                if track_id in tracked_objects:
                    tracked_obj = tracked_objects[track_id]
                    tracked_obj.bbox = bbox_dict
                    tracked_obj.confidence = confidence
                    tracked_obj.center_point = center
                    tracked_obj.trace.append(center)
                else:
                    tracked_obj = TrackedObject(
                        track_id=track_id,
                        class_id=0,  # person
                        confidence=confidence,
                        bbox=bbox_dict,
                        center_point=center
                    )
                    tracked_objects[track_id] = tracked_obj
                
                current_objects.append(tracked_obj)
            
            yield {
                "frame": frame,
                "tracked_objects": current_objects,
                "frame_number": frame_count
            }
            
    finally:
        cap.release()

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
    tracking_results: List[Dict],
    output_path: str = "./object/tests/fixtures/csv_object_tracking/tracking_results.csv"
) -> None:
    """
    Save tracking results to CSV format.
    
    Format:
    frame_number,track_id,class_id,confidence,x1,y1,x2,y2,center_x,center_y
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow([
            'frame_number',
            'track_id',
            'class_id',
            'confidence',
            'x1', 'y1', 'x2', 'y2',
            'center_x', 'center_y'
        ])
        
        # Write data
        for result in tracking_results:
            frame_number = result["frame_number"]
            for obj in result["tracked_objects"]:
                bbox = obj.bbox
                center_x, center_y = obj.center_point
                
                writer.writerow([
                    frame_number,
                    obj.track_id,
                    obj.class_id,
                    f"{obj.confidence:.3f}",
                    f"{bbox['x1']:.2f}",
                    f"{bbox['y1']:.2f}",
                    f"{bbox['x2']:.2f}",
                    f"{bbox['y2']:.2f}",
                    center_x,
                    center_y
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
                
            frame = result["frame"]
            tracked_objects = result["tracked_objects"]
            
            # Convert our tracked objects back to supervision format
            detections = sv.Detections(
                xyxy=np.array([list(obj.bbox.values()) for obj in tracked_objects]),
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