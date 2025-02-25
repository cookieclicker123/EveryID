import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import logging
import numpy as np
import os
import ssl

def setup_logging():
    logging.basicConfig(
        filename='download_debug.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def download_cifar10():
    # Fix SSL certificate issue on macOS
    if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
        getattr(ssl, '_create_unverified_context', None)):
        ssl._create_default_https_context = ssl._create_unverified_context
        logging.info("Created unverified HTTPS context to fix SSL certificate issue")
    
    # Create directories
    base_dir = Path("./object/vehicle_classifier/tmp")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Download via tensorflow
    logging.info("Downloading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    logging.info(f"Download complete. Training images: {len(x_train)}, Test images: {len(x_test)}")
    
    # CIFAR-10 class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
              'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Select only vehicle classes (0=airplane, 1=automobile, 8=ship, 9=truck)
    vehicle_indices = [0, 1, 8, 9]
    vehicle_names = [classes[i] for i in vehicle_indices]
    
    logging.info(f"Selected vehicle classes: {vehicle_names}")
    
    # Create YOLO directory structure
    images_dir = base_dir / 'images'
    labels_dir = base_dir / 'labels'
    
    for split in ['train', 'val']:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Process training data
    vehicle_count = 0
    for i, (image, label) in enumerate(zip(x_train, y_train)):
        if label[0] in vehicle_indices:
            # Save image
            img_path = images_dir / 'train' / f'vehicle_{i}.jpg'
            keras.utils.save_img(img_path, image)
            
            # Create YOLO label (assuming object takes up most of image)
            # Format: <class> <x_center> <y_center> <width> <height>
            class_idx = vehicle_indices.index(label[0])
            label_content = f"{class_idx} 0.5 0.5 0.8 0.8\n"
            
            label_path = labels_dir / 'train' / f'vehicle_{i}.txt'
            label_path.write_text(label_content)
            
            vehicle_count += 1
            if vehicle_count % 1000 == 0:
                logging.info(f"Processed {vehicle_count} training images")
    
    logging.info(f"Total training vehicles processed: {vehicle_count}")
    
    # Process test data for validation
    vehicle_count = 0
    for i, (image, label) in enumerate(zip(x_test, y_test)):
        if label[0] in vehicle_indices:
            img_path = images_dir / 'val' / f'vehicle_{i}.jpg'
            keras.utils.save_img(img_path, image)
            
            class_idx = vehicle_indices.index(label[0])
            label_content = f"{class_idx} 0.5 0.5 0.8 0.8\n"
            
            label_path = labels_dir / 'val' / f'vehicle_{i}.txt'
            label_path.write_text(label_content)
            
            vehicle_count += 1
    
    logging.info(f"Total validation vehicles processed: {vehicle_count}")

    # Create dataset.yaml
    yaml_content = f"""
path: {base_dir.absolute()}
train: images/train
val: images/val
names: {vehicle_names}
    """
    
    yaml_path = base_dir / 'dataset.yaml'
    yaml_path.write_text(yaml_content)
    
    logging.info("Dataset preparation completed")
    return base_dir / 'dataset.yaml'

if __name__ == "__main__":
    setup_logging()
    yaml_path = download_cifar10()
    print(f"Dataset prepared. YAML config at: {yaml_path}") 