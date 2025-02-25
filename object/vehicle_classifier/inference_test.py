from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch

def setup_logging():
    """Configure logging for inference process"""
    logging.basicConfig(
        filename='inference_debug.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("=== Starting Vehicle Classifier Inference ===")

def check_device():
    """Check and configure the best available device"""
    if torch.cuda.is_available():
        device = 'cuda'
        device_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        device_name = 'Apple Silicon GPU'
    else:
        device = 'cpu'
        device_name = 'CPU'
    
    print(f"Inference on: {device_name} ({device})")
    logging.info(f"Using device: {device_name} ({device})")
    return device

def load_model():
    """Load the best trained model"""
    model_path = Path("./reid/runs/detect/train/weights/best.pt")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please run training first.")
    
    model = YOLO(model_path)
    logging.info(f"Loaded model from {model_path}")
    return model

def run_inference(model, device):
    """Run inference on test images"""
    test_dir = Path("./object/vehicle_classifier/tmp/test_images")
    
    if not test_dir.exists():
        test_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created test images directory at {test_dir}")
        print(f"Please add test images to {test_dir}")
        return
    
    # Get all jpg images
    test_images = list(test_dir.glob("*.jpg"))
    
    if not test_images:
        logging.warning(f"No test images found in {test_dir}")
        print(f"No test images found. Please add .jpg images to {test_dir}")
        return
    
    logging.info(f"Found {len(test_images)} test images")
    
    # Create output directory
    output_dir = Path("./object/vehicle_classifier/tmp/inference_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure for visualization
    fig, axes = plt.subplots(len(test_images), 1, figsize=(10, 5*len(test_images)))
    if len(test_images) == 1:
        axes = [axes]
    
    # Process each image
    results = []
    for i, img_path in enumerate(test_images):
        logging.info(f"Processing {img_path.name}")
        
        # Run inference
        result = model(img_path, device=device, verbose=False)[0]
        results.append(result)
        
        # Get prediction
        if len(result.boxes) > 0:
            # Get highest confidence prediction
            conf = result.boxes.conf.max().item()
            cls_id = int(result.boxes.cls[result.boxes.conf.argmax()])
            cls_name = result.names[cls_id]
            
            prediction = f"{cls_name} ({conf:.2f})"
            color = 'green'  # Correct by default
            
            # Check if filename contains the class name
            expected_class = img_path.stem.lower()
            if expected_class == 'car':  # Handle car/automobile naming
                expected_class = 'automobile'
            if expected_class == 'plane':  # Handle plane/airplane naming
                expected_class = 'airplane'
                
            if cls_name.lower() != expected_class and expected_class in result.names.values():
                color = 'red'  # Incorrect prediction
                logging.warning(f"Incorrect prediction for {img_path.name}: expected {expected_class}, got {cls_name}")
        else:
            prediction = "No detection"
            color = 'red'
            logging.warning(f"No detection for {img_path.name}")
        
        # Display image and prediction
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(img)
        axes[i].set_title(f"File: {img_path.name} | Prediction: {prediction}", color=color)
        axes[i].axis('off')
        
        # Save result image
        result_img = result.plot()
        cv2.imwrite(str(output_dir / f"result_{img_path.name}"), result_img)
    
    # Save summary plot
    plt.tight_layout()
    plt.savefig(output_dir / "summary.png")
    logging.info(f"Saved summary to {output_dir / 'summary.png'}")
    
    # Display summary
    print("\nInference Results:")
    print("-----------------")
    for i, img_path in enumerate(test_images):
        result = results[i]
        if len(result.boxes) > 0:
            conf = result.boxes.conf.max().item()
            cls_id = int(result.boxes.cls[result.boxes.conf.argmax()])
            cls_name = result.names[cls_id]
            print(f"{img_path.name}: {cls_name} (confidence: {conf:.2f})")
        else:
            print(f"{img_path.name}: No detection")
    
    print(f"\nDetailed results saved to {output_dir}")

def main():
    """Main inference function"""
    setup_logging()
    
    try:
        device = check_device()
        model = load_model()
        run_inference(model, device)
        
    except Exception as e:
        logging.error(f"Inference failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 