from typing import Generator
import cv2
import sys
import os

# Add the project root to the path when running as a script
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultralytics import YOLO
import numpy as np
from object.data_model import Detection, FrameResult, ModelConfig, BoundingBox, VideoProcessor

def load_yolo_model() -> YOLO:
    """Load YOLO model from fixtures."""
    model_config = ModelConfig(model_path="./object/tests/fixtures/models/yolo11l.pt")
    return model_config.load_model()

class PersonDetectionProcessor(VideoProcessor):
    """Video processor for person detection."""
    
    def _process_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> FrameResult:
        """Process a single frame to detect people."""
        results = self.model.predict(frame, conf=self.model_config.conf_threshold)
        
        detections = []
        for box in results[0].boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = box
            # Only include person detections (class 0)
            if int(class_id) == 0:
                detections.append(Detection(
                    class_id=0,  # Always person
                    confidence=float(confidence),
                    bbox=BoundingBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2)
                    )
                ))
        
        return FrameResult(
            frame=frame,
            detections=detections,
            frame_number=frame_number,
            timestamp=timestamp
        )

def process_video_frames(
    video_path: str,
    model: YOLO,
    conf_threshold: float = 0.5,
    skip_frames: int = 0
) -> Generator[FrameResult, None, None]:
    """
    Process video frames and detect people (class 0).
    
    Args:
        video_path: Path to video file
        model: YOLO model instance
        conf_threshold: Confidence threshold for detections
        skip_frames: Number of frames to skip between detections (0 = process every frame)
    
    Yields:
        FrameResult containing frame and person detections
    """
    if conf_threshold <= 0 or conf_threshold > 1:
        raise ValueError("Confidence threshold must be between 0 and 1")
    if skip_frames < 0:
        raise ValueError("Skip frames must be non-negative")
    
    model_config = ModelConfig(
        model_path="",  # Not needed as model is already loaded
        conf_threshold=conf_threshold
    )
    processor = PersonDetectionProcessor(model_config)
    processor.model = model  # Use the provided model
    
    yield from processor.process_video(video_path, skip_frames)

def save_detection_video(
    video_path: str,
    output_path: str,
    model: YOLO,
    conf_threshold: float = 0.5,
    skip_frames: int = 0,
    show_preview: bool = False
) -> None:
    """
    Process video and save with detection boxes.
    
    Args:
        video_path: Path to input video
        output_path: Path to save processed video
        model: YOLO model instance
        conf_threshold: Confidence threshold for detections
        skip_frames: Number of frames to skip between detections
        show_preview: Whether to show preview window while processing
    """
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
    
    try:
        for result in process_video_frames(video_path, model, conf_threshold, skip_frames):
            frame = result.frame
            
            # Draw detection boxes
            for det in result.detections:
                bbox = det.bbox
                x1, y1 = int(bbox.x1), int(bbox.y1)
                x2, y2 = int(bbox.x2), int(bbox.y2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Class {det.class_id}: {det.confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
            
            out.write(frame)
            
            if show_preview:
                cv2.imshow('Detection Preview', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
    finally:
        out.release()
        if show_preview:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    model = load_yolo_model()
    video_path = "./object/tests/fixtures/videos/sample.mp4"
    output_path = "./object/tests/fixtures/videos/sample_detected.mp4"
    
    save_detection_video(
        video_path=video_path,
        output_path=output_path,
        model=model,
        conf_threshold=0.5,
        skip_frames=2,  # Process every 3rd frame
        show_preview=True
    ) 