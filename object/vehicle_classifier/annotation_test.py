import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import logging
import random

def setup_logging():
    logging.basicConfig(
        filename='annotation_debug.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def verify_dataset():
    # Load dataset config
    base_dir = Path("./object/vehicle_classifier/tmp")
    yaml_path = base_dir / 'dataset.yaml'
    
    if not yaml_path.exists():
        raise FileNotFoundError("Dataset YAML not found. Run download_test.py first.")
    
    with open(yaml_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    class_names = dataset_config['names']
    logging.info(f"Dataset classes: {class_names}")
    
    # Check directory structure
    train_img_dir = base_dir / 'images' / 'train'
    train_label_dir = base_dir / 'labels' / 'train'
    val_img_dir = base_dir / 'images' / 'val'
    val_label_dir = base_dir / 'labels' / 'val'
    
    # Count files by class
    train_counts = {cls: 0 for cls in class_names}
    val_counts = {cls: 0 for cls in class_names}
    
    # Check image sizes
    image_sizes = []
    
    # Verify training set
    logging.info("Verifying training set...")
    for img_path in train_img_dir.glob('*.jpg'):
        label_path = train_label_dir / f"{img_path.stem}.txt"
        
        # Check if label exists
        if not label_path.exists():
            logging.warning(f"Missing label for {img_path}")
            continue
        
        # Read label
        with open(label_path, 'r') as f:
            label_content = f.read().strip()
        
        class_idx = int(label_content.split()[0])
        if class_idx < len(class_names):
            train_counts[class_names[class_idx]] += 1
        
        # Check image size
        with Image.open(img_path) as img:
            image_sizes.append(img.size)
    
    # Verify validation set
    logging.info("Verifying validation set...")
    for img_path in val_img_dir.glob('*.jpg'):
        label_path = val_label_dir / f"{img_path.stem}.txt"
        
        # Check if label exists
        if not label_path.exists():
            logging.warning(f"Missing label for {img_path}")
            continue
        
        # Read label
        with open(label_path, 'r') as f:
            label_content = f.read().strip()
        
        class_idx = int(label_content.split()[0])
        if class_idx < len(class_names):
            val_counts[class_names[class_idx]] += 1
    
    # Log statistics
    logging.info("Dataset Statistics:")
    logging.info(f"Training images: {sum(train_counts.values())}")
    logging.info(f"Validation images: {sum(val_counts.values())}")
    
    for cls in class_names:
        logging.info(f"Class '{cls}': {train_counts[cls]} train, {val_counts[cls]} val")
    
    # Check image sizes
    unique_sizes = set(image_sizes)
    logging.info(f"Image sizes: {unique_sizes}")
    
    # Visualize random samples
    visualize_samples(base_dir, class_names)
    
    return {
        'train_counts': train_counts,
        'val_counts': val_counts,
        'image_sizes': unique_sizes
    }

def visualize_samples(base_dir, class_names):
    """Visualize random samples from each class"""
    train_img_dir = base_dir / 'images' / 'train'
    train_label_dir = base_dir / 'labels' / 'train'
    
    # Create output directory
    samples_dir = base_dir / 'samples'
    samples_dir.mkdir(exist_ok=True)
    
    # Group images by class
    class_images = {i: [] for i in range(len(class_names))}
    
    for img_path in train_img_dir.glob('*.jpg'):
        label_path = train_label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path, 'r') as f:
                class_idx = int(f.read().strip().split()[0])
                class_images[class_idx].append(img_path)
    
    # Create a figure with samples from each class
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for i, class_idx in enumerate(range(len(class_names))):
        if class_images[class_idx]:
            # Select random samples
            samples = random.sample(class_images[class_idx], min(5, len(class_images[class_idx])))
            
            # Create a subplot for this class
            ax = axes[i]
            
            # Display first sample
            sample_path = samples[0]
            img = np.array(Image.open(sample_path))
            ax.imshow(img)
            ax.set_title(f"{class_names[class_idx]} (n={len(class_images[class_idx])})")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(samples_dir / 'class_samples.png')
    logging.info(f"Sample visualization saved to {samples_dir / 'class_samples.png'}")
    
    # Create a grid of more samples
    plt.figure(figsize=(15, 10))
    for class_idx in range(len(class_names)):
        if class_images[class_idx]:
            samples = random.sample(class_images[class_idx], min(10, len(class_images[class_idx])))
            for j, sample_path in enumerate(samples[:5]):
                plt.subplot(len(class_names), 5, class_idx*5 + j + 1)
                img = np.array(Image.open(sample_path))
                plt.imshow(img)
                if j == 0:
                    plt.ylabel(class_names[class_idx])
                plt.xticks([])
                plt.yticks([])
    
    plt.tight_layout()
    plt.savefig(samples_dir / 'more_samples.png')
    logging.info(f"Extended sample visualization saved to {samples_dir / 'more_samples.png'}")

if __name__ == "__main__":
    setup_logging()
    stats = verify_dataset()
    print("Dataset verification complete. See annotation_debug.log for details.")
    print(f"Training images: {sum(stats['train_counts'].values())}")
    print(f"Validation images: {sum(stats['val_counts'].values())}")
    print(f"Image sizes: {stats['image_sizes']}") 