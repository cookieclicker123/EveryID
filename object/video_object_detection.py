from typing import List, Dict, Union, Generator
import cv2
from ultralytics import YOLO
import numpy as np

def load_yolo_model() -> YOLO:
    """Load YOLO model from fixtures."""
    model = YOLO("./object/tests/fixtures/models/yolo11l.pt")
    return model

def process_video_frames(
    video_path: str,
    model: YOLO,
    conf_threshold: float = 0.5,
    skip_frames: int = 0
) -> Generator[Dict[str, Union[np.ndarray, List[Dict]]], None, None]:
    """
    Process video frames and detect people (class 0).
    
    Args:
        video_path: Path to video file
        model: YOLO model instance
        conf_threshold: Confidence threshold for detections
        skip_frames: Number of frames to skip between detections (0 = process every frame)
    
    Yields:
        Dictionary containing frame and person detections
    """
    if conf_threshold <= 0 or conf_threshold > 1:
        raise ValueError("Confidence threshold must be between 0 and 1")
    if skip_frames < 0:
        raise ValueError("Skip frames must be non-negative")
        
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
                
            results = model.predict(frame, conf=conf_threshold)
            
            detections = []
            for box in results[0].boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = box
                # Only include person detections (class 0)
                if int(class_id) == 0:
                    detections.append({
                        "class_id": 0,  # Always person
                        "confidence": float(confidence),
                        "bbox": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2)
                        }
                    })
            
            yield {
                "frame": frame,
                "detections": detections,
                "frame_number": frame_count
            }
            
    finally:
        cap.release()

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
            frame = result["frame"]
            detections = result["detections"]
            
            # Draw detection boxes
            for det in detections:
                bbox = det["bbox"]
                x1, y1 = int(bbox["x1"]), int(bbox["y1"])
                x2, y2 = int(bbox["x2"]), int(bbox["y2"])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Class {det['class_id']}: {det['confidence']:.2f}",
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