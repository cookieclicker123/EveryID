import os
from pathlib import Path
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import scipy.io as sio
import shutil

def test_dataset_download():
    """Test downloading and checking the Stanford Cars Dataset"""
    print("\nStarting Stanford Cars Dataset download test...")
    
    # Initialize Kaggle API
    try:
        api = KaggleApi()
        api.authenticate()
        print("✓ Kaggle API authenticated")
    except Exception as e:
        print(f"✗ Kaggle API authentication failed: {e}")
        return False

    # Create tmp directory under car_classifier
    base_dir = Path("./object/car_classifier/tmp")
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {base_dir}")

    # Clean existing data if any
    for item in base_dir.glob('*'):
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    print("✓ Cleaned existing data")

    # Download dataset
    try:
        print("\nDownloading Stanford Cars Dataset...")
        api.dataset_download_files(
            'jessicali9530/stanford-cars-dataset',
            path=base_dir,
            unzip=True
        )
        print("✓ Download completed")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False

    # Check all three critical components
    print("\nVerifying dataset components:")
    
    # 1. Check cars_annos.mat
    annos_path = base_dir / 'cars_annos.mat'
    if annos_path.exists():
        try:
            annotations = sio.loadmat(str(annos_path))
            print(f"✓ Found and loaded cars_annos.mat")
            print(f"  - Annotation keys: {list(annotations.keys())}")
        except Exception as e:
            print(f"✗ Error reading cars_annos.mat: {e}")
            return False
    else:
        print("✗ Missing cars_annos.mat")
        return False

    # 2. Check training images
    train_path = base_dir / 'cars_train' / 'cars_train'
    train_images = list(train_path.glob('*.jpg'))
    print(f"✓ Training images found: {len(train_images)}")
    
    # 3. Check testing images
    test_path = base_dir / 'cars_test' / 'cars_test'
    test_images = list(test_path.glob('*.jpg'))
    print(f"✓ Testing images found: {len(test_images)}")

    return True

if __name__ == "__main__":
    success = test_dataset_download()
    if success:
        print("\n✓ Dataset verification successful!")
    else:
        print("\n✗ Dataset verification failed!") 