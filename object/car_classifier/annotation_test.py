import scipy.io as sio
from pathlib import Path
import numpy as np

def convert_filename(original_path):
    """Convert from annotation format (000001.jpg) to actual format (00001.jpg)"""
    filename = Path(original_path).name
    number = int(filename.replace('.jpg', ''))
    return f"{number:05d}.jpg"

def test_annotations():
    """Test and explore the Stanford Cars Dataset annotations"""
    print("\nAnalyzing Stanford Cars Dataset annotations...")
    
    base_dir = Path("./object/car_classifier/tmp")  # Updated to use tmp directory
    annos_path = base_dir / 'cars_annos.mat'
    
    try:
        # Load annotations
        data = sio.loadmat(str(annos_path))
        annotations = data['annotations'][0]
        class_names = data['class_names'][0]
        
        # Basic dataset info
        print(f"\nDataset Overview:")
        print(f"Total annotations: {len(annotations)}")
        print(f"Total classes: {len(class_names)}")
        
        # Examine annotation structure
        print("\nAnnotation Structure:")
        sample_anno = annotations[0]
        print(f"Fields in each annotation: {sample_anno.dtype.names}")
        
        # Show first few annotations in detail
        print("\nSample Annotations (first 5):")
        for i, anno in enumerate(annotations[:5]):
            print(f"\nAnnotation {i+1}:")
            for field in anno.dtype.names:
                print(f"{field}: {anno[field][0]}")
        
        # Show first few class names
        print("\nSample Classes (first 5):")
        for i, class_name in enumerate(class_names[:5]):
            print(f"{i+1}. {class_name[0].strip()}")
        
        # Count train/test split
        train_count = sum(1 for anno in annotations if not bool(anno['test'][0][0]))
        test_count = sum(1 for anno in annotations if bool(anno['test'][0][0]))
        print(f"\nSplit in annotations:")
        print(f"Training images: {train_count}")
        print(f"Testing images: {test_count}")
        
        # Verify these numbers against actual files
        train_dir = base_dir / 'cars_train' / 'cars_train'
        test_dir = base_dir / 'cars_test' / 'cars_test'
        
        actual_train = len(list(train_dir.glob('*.jpg')))
        actual_test = len(list(test_dir.glob('*.jpg')))
        print(f"\nActual files in directories:")
        print(f"Training images: {actual_train}")
        print(f"Testing images: {actual_test}")
        
        # Show sample filenames from both annotations and directories
        print("\nSample paths comparison:")
        print("From annotations:")
        for anno in annotations[:3]:
            print(f"  {anno['relative_im_path'][0]}")
        
        print("\nFrom directories:")
        print("Training:")
        for f in sorted(train_dir.glob('*.jpg'))[:3]:
            print(f"  {f.name}")
        print("Testing:")
        for f in sorted(test_dir.glob('*.jpg'))[:3]:
            print(f"  {f.name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error analyzing annotations: {e}")
        return False

if __name__ == "__main__":
    success = test_annotations()
    if success:
        print("\n✓ Annotation analysis successful!")
    else:
        print("\n✗ Annotation analysis failed!") 