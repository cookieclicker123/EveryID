from ultralytics import YOLO
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import yaml
import time
import torch

def setup_logging():
    """Configure logging for training process"""
    logging.basicConfig(
        filename='train_debug.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("=== Starting Vehicle Classifier Training ===")

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
    
    print(f"Training on: {device_name} ({device})")
    logging.info(f"Using device: {device_name} ({device})")
    return device

def load_dataset_info():
    """Load and verify dataset configuration"""
    base_dir = Path("./object/vehicle_classifier/tmp")
    yaml_path = base_dir / 'dataset.yaml'
    
    if not yaml_path.exists():
        raise FileNotFoundError("Dataset YAML not found. Run download_test.py first.")
    
    with open(yaml_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    class_names = dataset_config['names']
    logging.info(f"Training on classes: {class_names}")
    logging.info(f"Dataset path: {dataset_config['path']}")
    
    return yaml_path, class_names

def train_model(yaml_path, device):
    """Train YOLO model on vehicle dataset"""
    # Initialize model - using smaller variant since images are 32x32
    model = YOLO('yolov8n.pt')
    logging.info(f"Initialized model: YOLOv8n ({model.model.yaml['nc']} classes → 4 classes)")
    
    # Configure training parameters optimized for this dataset
    start_time = time.time()
    results = model.train(
        data=yaml_path,
        epochs=20,           # Fewer epochs needed for this dataset
        imgsz=32,            # CIFAR-10 image size
        batch=128,           # Larger batch size for small images
        patience=5,          # Early stopping if no improvement
        lr0=0.001,           # Initial learning rate
        lrf=0.01,            # Final learning rate factor
        warmup_epochs=2.0,   # Warmup period
        cos_lr=True,         # Cosine learning rate schedule
        weight_decay=0.0005, # L2 regularization
        box=0.05,            # Box loss weight (reduced since detection is easier)
        cls=0.5,             # Classification loss weight
        dfl=1.5,             # Distribution focal loss weight
        plots=True,          # Generate plots
        save=True,           # Save checkpoints
        device=device,       # Use detected device
        verbose=True         # Show progress
    )
    
    training_time = time.time() - start_time
    logging.info(f"Training completed in {training_time:.1f} seconds")
    return results, model

def evaluate_model(model):
    """Evaluate model performance"""
    base_dir = Path("./object/vehicle_classifier/tmp")
    val_dir = base_dir / 'images' / 'val'
    
    logging.info("Running validation...")
    results = model.val()
    
    metrics = {
        'mAP50': results.box.map50,
        'mAP50-95': results.box.map,
        'precision': results.box.p,
        'recall': results.box.r
    }
    
    logging.info(f"Validation metrics: {metrics}")
    return metrics

def main():
    """Main training function"""
    start_time = time.time()
    setup_logging()
    
    try:
        # Check for MPS/GPU availability
        device = check_device()
        
        yaml_path, class_names = load_dataset_info()
        results, model = train_model(yaml_path, device)
        metrics = evaluate_model(model)
        
        # Log final results
        logging.info("\n=== Training Summary ===")
        logging.info(f"Classes: {class_names}")
        logging.info(f"Final mAP50: {metrics['mAP50']:.4f}")
        logging.info(f"Final mAP50-95: {metrics['mAP50-95']:.4f}")
        logging.info(f"Total training time: {(time.time() - start_time)/60:.1f} minutes")
        
        print(f"\nTraining completed successfully!")
        print(f"Model saved to: {Path('./object/vehicle_classifier/tmp/runs/detect/train')}")
        print(f"mAP50: {metrics['mAP50']:.4f}")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 