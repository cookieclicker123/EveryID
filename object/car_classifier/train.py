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
        """Process and convert annotations to YOLO format"""
        print("\nProcessing split:", split_name)
        processed = 0
        failed = 0
        
        # Debug first few conversions
        print("\nChecking file mappings:")
        for idx in indices[:5]:
            anno = train_annos[idx]
            orig_path = str(anno['relative_im_path'][0])
            src_name = convert_filename(orig_path.replace('car_ims/', ''))
            actual_path = train_dir / src_name
            print(f"Original: {orig_path} -> Converted: {src_name} -> Exists: {actual_path.exists()}")
        
        # Continue with processing...
        for idx in indices:
            anno = train_annos[idx]
            try:
                orig_path = str(anno['relative_im_path'][0])
                src_name = convert_filename(orig_path.replace('car_ims/', ''))
                
                src_path = train_dir / src_name
                if src_path.exists():
                    dst_path = dataset_dir / 'images' / split_name / src_name
                    shutil.copy(src_path, dst_path)
                    
                    # Get current image dimensions
                    with Image.open(src_path) as img:
                        current_size = img.size
                    
                    # Get original bbox coordinates
                    x1 = float(anno['bbox_x1'][0][0])
                    y1 = float(anno['bbox_y1'][0][0])
                    x2 = float(anno['bbox_x2'][0][0])
                    y2 = float(anno['bbox_y2'][0][0])
                    
                    # Scale coordinates to match current image size
                    x1, y1, x2, y2 = scale_coordinates(
                        x1, y1, x2, y2,
                        (3220, 2506),  # Max bbox dimensions from annotations
                        current_size
                    )
                    
                    if processed < 5:  # Debug first 5
                        print(f"\nProcessing {src_name}:")
                        print(f"Image size: {current_size}")
                        print(f"Scaled bbox: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
                    
                    # Normalize coordinates
                    x1 = x1 / current_size[0]
                    y1 = y1 / current_size[1]
                    x2 = x2 / current_size[0]
                    y2 = y2 / current_size[1]
                    
                    if processed < 5:
                        print(f"Normalized: x1={x1:.3f}, y1={y1:.3f}, x2={x2:.3f}, y2={y2:.3f}")
                    
                    # Convert to YOLO format
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Create label file
                    label_path = dataset_dir / 'labels' / split_name / src_name.replace('.jpg', '.txt')
                    with open(label_path, 'w') as f:
                        class_id = int(anno['class'][0][0]) - 1
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                    
                    processed += 1
                    
            except Exception as e:
                print(f"Error processing {src_name}: {str(e)}")
                failed += 1
        
        print(f"\n{split_name} Split Summary:")
        print(f"Processed successfully: {processed}")
        print(f"Failed to process: {failed}")
        print(f"Coverage: {(processed/len(train_annos))*100:.1f}%")
        
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
    model = YOLO('yolov8l.pt')
    
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