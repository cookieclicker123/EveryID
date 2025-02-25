import torch
import torch.optim as optim
from pathlib import Path
import scipy.io as sio
import numpy as np
from ultralytics import YOLO
import shutil
from sklearn.model_selection import train_test_split
import yaml
from PIL import Image
import logging

def verify_device():
    """Verify and setup the fastest available device"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print(f"Using CPU")
    return device

def scale_coordinates(x1, y1, x2, y2, orig_size, new_size):
    """Scale coordinates from original image size to new size"""
    x_scale = new_size[0] / orig_size[0]
    y_scale = new_size[1] / orig_size[1]
    
    # Scale coordinates
    x1 = x1 * x_scale
    y1 = y1 * y_scale
    x2 = x2 * x_scale
    y2 = y2 * y_scale
    
    # Clamp to image boundaries
    x1 = max(0, min(x1, new_size[0]))
    y1 = max(0, min(y1, new_size[1]))
    x2 = max(0, min(x2, new_size[0]))
    y2 = max(0, min(y2, new_size[1]))
    
    return x1, y1, x2, y2

def convert_filename(original_path):
    """Convert between annotation format (000001.jpg) and actual format (00001.jpg)"""
    filename = Path(original_path).name
    number = int(filename.replace('.jpg', ''))
    
    # Map to sequential numbering (1-8144 for training)
    # The actual files are numbered sequentially from 00001.jpg to 08144.jpg
    return f"{number%8144+1:05d}.jpg"  # Use modulo to wrap around to valid range

def prepare_data():
    """Prepare dataset structure for YOLO training"""
    print("\nPreparing dataset for training...")
    
    base_dir = Path("./object/car_classifier/tmp")
    train_dir = base_dir / 'cars_train' / 'cars_train'
    annos_path = base_dir / 'cars_annos.mat'
    
    # Create YOLO dataset structure
    dataset_dir = base_dir / 'yolo_dataset'
    dataset_dir.mkdir(exist_ok=True)
    
    # Load annotations
    data = sio.loadmat(str(annos_path))
    annotations = data['annotations'][0]
    class_names = [name[0].strip() for name in data['class_names'][0]]
    
    # Create class mapping file
    with open(dataset_dir / 'classes.txt', 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    # Get training images (test=0)
    train_annos = [anno for anno in annotations if not bool(anno['test'][0][0])]
    
    print(f"Total annotations: {len(annotations)}")
    print(f"Training annotations: {len(train_annos)}")
    print(f"First few training paths:")
    for anno in train_annos[:5]:
        print(f"  {anno['relative_im_path'][0]}")
    
    # Split into train and val
    train_indices, val_indices = train_test_split(
        range(len(train_annos)), 
        test_size=0.1, 
        random_state=42
    )
    
    # Create dataset.yaml
    dataset_yaml = {
        'path': str(dataset_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(dataset_dir / 'dataset.yaml', 'w') as f:
        yaml.dump(dataset_yaml, f)
    
    # Create directories
    (dataset_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (dataset_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (dataset_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (dataset_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # Process annotations and copy images
    def analyze_annotations(annotations):
        """Analyze the annotation format and ranges"""
        print("\nAnalyzing annotation statistics:")
        
        x1_vals = [float(anno['bbox_x1'][0][0]) for anno in annotations]
        y1_vals = [float(anno['bbox_y1'][0][0]) for anno in annotations]
        x2_vals = [float(anno['bbox_x2'][0][0]) for anno in annotations]
        y2_vals = [float(anno['bbox_y2'][0][0]) for anno in annotations]
        
        print(f"X1 range: {min(x1_vals):.2f} to {max(x1_vals):.2f}")
        print(f"Y1 range: {min(y1_vals):.2f} to {max(y1_vals):.2f}")
        print(f"X2 range: {min(x2_vals):.2f} to {max(x2_vals):.2f}")
        print(f"Y2 range: {min(y2_vals):.2f} to {max(y2_vals):.2f}")
        
        return max(x1_vals), max(y1_vals), max(x2_vals), max(y2_vals)

    def process_split(indices, split_name):
        processed = 0
        failed = 0
        size_mismatches = 0
        
        logging.info(f"\nProcessing {split_name} split")
        logging.info(f"Total indices to process: {len(indices)}")
        
        for idx in indices:
            anno = train_annos[idx]
            try:
                orig_path = str(anno['relative_im_path'][0])
                src_name = convert_filename(orig_path.replace('car_ims/', ''))
                src_path = train_dir / src_name
                
                if src_path.exists():
                    with Image.open(src_path) as img:
                        width, height = img.size
                        
                        # Get bbox coordinates
                        x1 = float(anno['bbox_x1'][0][0])
                        y1 = float(anno['bbox_y1'][0][0])
                        x2 = float(anno['bbox_x2'][0][0])
                        y2 = float(anno['bbox_y2'][0][0])
                        
                        # Calculate center coordinates and dimensions (YOLO format)
                        # YOLO wants: <center-x> <center-y> <width> <height>
                        # All values normalized between 0 and 1
                        box_width = x2 - x1
                        box_height = y2 - y1
                        center_x = x1 + (box_width / 2)
                        center_y = y1 + (box_height / 2)
                        
                        # Normalize by image dimensions
                        center_x /= width
                        center_y /= height
                        box_width /= width
                        box_height /= height
                        
                        # Clamp values between 0 and 1
                        center_x = max(0, min(1, center_x))
                        center_y = max(0, min(1, center_y))
                        box_width = max(0, min(1, box_width))
                        box_height = max(0, min(1, box_height))
                        
                        # Log for debugging
                        logging.debug(f"\nProcessing {src_name}:")
                        logging.debug(f"Image size: {width}x{height}")
                        logging.debug(f"Original bbox: ({x1},{y1}) to ({x2},{y2})")
                        logging.debug(f"Normalized YOLO format: {center_x:.3f} {center_y:.3f} {box_width:.3f} {box_height:.3f}")
                        
                        # Only process if box makes sense
                        if 0 < box_width < 1 and 0 < box_height < 1:
                            processed += 1
                        else:
                            size_mismatches += 1
                            logging.warning(f"Invalid box dimensions for {src_name}")
                            
                else:
                    logging.error(f"File not found: {src_path}")
                    failed += 1
                
            except Exception as e:
                logging.error(f"Error processing {idx}: {str(e)}")
                failed += 1
        
        logging.info(f"\nSplit Summary for {split_name}:")
        logging.info(f"Total expected: {len(indices)}")
        logging.info(f"Processed: {processed}")
        logging.info(f"Failed: {failed}")
        logging.info(f"Size mismatches: {size_mismatches}")
        
        return processed, failed
    
    print("Processing training split...")
    process_split(train_indices, 'train')
    print("Processing validation split...")
    process_split(val_indices, 'val')
    
    return dataset_dir / 'dataset.yaml'

def train():
    """Train YOLO model on Stanford Cars Dataset"""
    print("\nStarting training...")
    
    # Setup device
    device = verify_device()
    
    base_dir = Path("./object/car_classifier/tmp")
    dataset_yaml = prepare_data()
    
    # Initialize model
    model = YOLO('yolo11l.pt')
    
    # Training arguments
    args = {
        'data': str(dataset_yaml),
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'device': device,
        'workers': 8,
        'project': str(base_dir / 'runs'),
        'name': 'car_classifier',
        'pretrained': True,
        'optimizer': 'Adam',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'patience': 50,
        'save': True,
        'save_period': -1,
        'cache': False,
        'exist_ok': True,
        'plots': True
    }
    
    # Train the model
    results = model.train(**args)
    print("\nTraining completed!")
    
    return results

if __name__ == "__main__":
    train() 