from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Generator
from enum import Enum
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import cv2
from collections import deque

class MovementSpeed(Enum):
    """Classification of movement speed."""
    STATIONARY = "stationary"
    WALKING = "walking"
    RUNNING = "running"

@dataclass
class BoundingBox:
    """Bounding box coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def width(self) -> float:
        """Get width of bounding box."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """Get height of bounding box."""
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @classmethod
    def from_dict(cls, bbox_dict: Dict[str, float]) -> "BoundingBox":
        """Create BoundingBox from dictionary."""
        return cls(
            x1=bbox_dict["x1"],
            y1=bbox_dict["y1"],
            x2=bbox_dict["x2"],
            y2=bbox_dict["y2"]
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2
        }
    
    def to_xyxy(self) -> List[float]:
        """Convert to [x1, y1, x2, y2] format."""
        return [self.x1, self.y1, self.x2, self.y2]

@dataclass
class Detection:
    """Object detection result."""
    class_id: int
    confidence: float
    bbox: BoundingBox
    
    @classmethod
    def from_yolo_result(cls, result: Any, idx: int) -> "Detection":
        """Create Detection from YOLO result."""
        boxes = result.boxes
        x1, y1, x2, y2 = boxes.xyxy[idx].tolist()
        class_id = int(boxes.cls[idx].item())
        confidence = float(boxes.conf[idx].item())
        
        return cls(
            class_id=class_id,
            confidence=confidence,
            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        )
    
    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "Detection":
        """Create Detection from dictionary."""
        return cls(
            class_id=detection_dict["class_id"],
            confidence=detection_dict["confidence"],
            bbox=BoundingBox.from_dict(detection_dict["bbox"])
        )

@dataclass
class TrackedObject:
    """Data class for tracked object information."""
    track_id: int
    class_id: int
    confidence: float
    bbox: BoundingBox
    center_point: Tuple[int, int]
    velocity: Optional[Tuple[float, float]] = None
    speed: Optional[float] = None  # Magnitude of velocity
    direction: Optional[float] = None  # Angle in degrees
    movement_type: Optional[MovementSpeed] = None
    last_timestamp: Optional[float] = None
    trace: deque = None
    
    def __post_init__(self):
        if self.trace is None:
            self.trace = deque(maxlen=30)  # Default trace length
        self.trace.append(self.center_point)
        
        # Convert bbox from dict if needed
        if isinstance(self.bbox, dict):
            self.bbox = BoundingBox.from_dict(self.bbox)

    def update_velocity(self, new_center: Tuple[int, int], current_timestamp: float) -> None:
        """Calculate velocity and derived movement metrics."""
        if self.last_timestamp is not None:
            time_diff = current_timestamp - self.last_timestamp
            if time_diff > 0:
                # Calculate velocity components
                dx = new_center[0] - self.center_point[0]
                dy = new_center[1] - self.center_point[1]
                self.velocity = (dx / time_diff, dy / time_diff)
                
                # Calculate speed (magnitude of velocity)
                self.speed = np.sqrt(dx**2 + dy**2) / time_diff
                
                # Calculate direction in degrees (0° is right, 90° is up)
                self.direction = np.degrees(np.arctan2(-dy, dx)) % 360
                
                # Classify movement
                if self.speed < 50:  # pixels per second
                    self.movement_type = MovementSpeed.STATIONARY
                elif self.speed < 200:
                    self.movement_type = MovementSpeed.WALKING
                else:
                    self.movement_type = MovementSpeed.RUNNING
        
        self.last_timestamp = current_timestamp

@dataclass
class FrameResult:
    """Result of processing a single video frame."""
    frame: np.ndarray
    detections: List[Detection] = field(default_factory=list)
    tracked_objects: List[TrackedObject] = field(default_factory=list)
    frame_number: int = 0
    timestamp: float = 0.0
    
    @property
    def has_detections(self) -> bool:
        """Check if there are any detections."""
        return len(self.detections) > 0
    
    @property
    def has_tracked_objects(self) -> bool:
        """Check if there are any tracked objects."""
        return len(self.tracked_objects) > 0

@dataclass
class ModelConfig:
    """Configuration for YOLO model."""
    model_path: str
    conf_threshold: float = 0.5
    device: str = "auto"
    
    def load_model(self) -> YOLO:
        """Load YOLO model with this configuration."""
        return YOLO(self.model_path)

class VideoProcessor:
    """Base class for video processing."""
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.model = model_config.load_model()
    
    def process_video(self, video_path: str, skip_frames: int = 0) -> Generator[FrameResult, None, None]:
        """Process video frames."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if skip_frames and frame_count % (skip_frames + 1) != 0:
                    continue
                
                timestamp = frame_count / fps
                
                # Process frame (to be implemented by subclasses)
                result = self._process_frame(frame, frame_count, timestamp)
                
                yield result
                
        finally:
            cap.release()
    
    def _process_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> FrameResult:
        """Process a single frame (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _process_frame") 