import os
import random
from pathlib import Path
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
import torch

def setup_model_and_classes():
    """Load the transformer model and define class names"""
    # Path to the model
    model_dir = Path("./tmp/scene_models/transformer")
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found at {model_dir}. Please run download_scene_transformer.py first.")
    
    # Check if all required files exist
    required_files = ["config.json", "model.safetensors", "preprocessor_config.json"]
    for file in required_files:
        if not (model_dir / file).exists():
            raise FileNotFoundError(f"Required model file {file} not found in {model_dir}")
    
    # Load the model
    try:
        print(f"Loading transformer model from {model_dir}...")
        
        # Load the image processor
        processor = ViTImageProcessor.from_pretrained(str(model_dir))
        
        # Load the model
        model = ViTForImageClassification.from_pretrained(str(model_dir))
        
        # Get class names from model config if available
        if hasattr(model.config, 'id2label') and model.config.id2label:
            class_names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
            print(f"Using {len(class_names)} class names from model config")
        else:
            # Fallback to default class names
            class_names = [
                'bathroom', 'beach', 'bedroom', 'classroom', 'field', 'forest', 
                'highway', 'kitchen', 'lake', 'library', 'lobby', 'mountain', 
                'office', 'park', 'playground', 'restaurant'
            ]
            print(f"Using {len(class_names)} default class names")
        
        print("Successfully loaded transformer model!")
        
        # Print model information
        print(f"Model has {model.config.num_labels} output classes")
        
        return model, processor, class_names
        
    except Exception as e:
        print(f"Error loading transformer model: {e}")
        raise

def preprocess_image(processor, img_path):
    """Preprocess an image for the transformer model"""
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        return inputs, image
    except Exception as e:
        print(f"Error preprocessing image {img_path}: {e}")
        return None, None

def predict_image(model, processor, img_path, class_names):
    """Make a prediction for an image"""
    try:
        inputs, img = preprocess_image(processor, img_path)
        if inputs is None:
            return None
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Print shape information for debugging
            print(f"Logits shape: {logits.shape}")
            print(f"Number of class names: {len(class_names)}")
            
            # Make sure we don't try to access more classes than we have
            num_classes = min(len(class_names), logits.shape[1])
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            
            # Get top predictions (up to 3, but limited by number of classes)
            top_k = min(3, num_classes)
            top_probs, top_indices = torch.topk(probabilities, top_k)
            top_probs = top_probs.cpu().numpy()
            top_indices = top_indices.cpu().numpy()
            
            # Map indices to class names
            top_classes = [class_names[i] for i in top_indices]
            
            # Ensure we have 3 classes and probabilities for consistent output
            # Pad with placeholders if needed
            while len(top_classes) < 3:
                top_classes.append("unknown")
            while len(top_probs) < 3:
                top_probs = np.append(top_probs, 0.0)
            
            return {
                'top_class': top_classes[0],
                'top_prob': top_probs[0],
                'top3_classes': top_classes,
                'top3_probs': top_probs,
                'img': img
            }
    except Exception as e:
        print(f"Error predicting image {img_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_inference_on_test_images():
    """Run inference on a set of test images"""
    # Setup model
    model, processor, class_names = setup_model_and_classes()
    
    # Check for test images directory
    test_dir = Path("./tmp/test_scene")
    if not test_dir.exists():
        print(f"Test directory {test_dir} not found. Creating it...")
        test_dir.mkdir(parents=True, exist_ok=True)
        print(f"Please add test images to {test_dir} and run again.")
        return []
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(test_dir.glob(f"*{ext}")))
    
    if not image_files:
        print(f"No images found in {test_dir}. Please add some images and run again.")
        return []
    
    print(f"Found {len(image_files)} images in {test_dir}")
    
    # Create output directory
    output_dir = Path("./tmp/scene_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images
    results = []
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            prediction = predict_image(model, processor, img_path, class_names)
            
            if prediction is None:
                continue
            
            # Store result
            result = {
                'path': img_path,
                'filename': img_path.name,
                'prediction': prediction['top_class'],
                'confidence': prediction['top_prob'],
                'top3_classes': prediction['top3_classes'],
                'top3_probs': prediction['top3_probs']
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Visualize results
    visualize_results(results, output_dir)
    
    # Print summary statistics
    print_summary_statistics(results)
    
    return results

def visualize_results(results, output_dir):
    """Visualize all the results"""
    if not results:
        print("No results to visualize.")
        return
    
    # Determine grid size based on number of images
    num_images = len(results)
    cols = min(5, num_images)
    rows = (num_images + cols - 1) // cols  # Ceiling division
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    
    # Handle case with only one image
    if num_images == 1:
        axes = np.array([axes])
    
    # Flatten axes for easy iteration
    axes = axes.flatten()
    
    for i, result in enumerate(results):
        if i >= len(axes):
            break
            
        img_path = result['path']
        
        try:
            # Load and display image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(img)
            
            # Format confidence as percentage
            confidence = result['confidence'] * 100
            
            # Display top 3 predictions
            title = f"File: {result['filename']}\n"
            title += f"Top: {result['prediction']} ({confidence:.1f}%)\n"
            title += f"2nd: {result['top3_classes'][1]} ({result['top3_probs'][1]*100:.1f}%)\n"
            title += f"3rd: {result['top3_classes'][2]} ({result['top3_probs'][2]*100:.1f}%)"
            
            axes[i].set_title(title, fontsize=10)
            axes[i].axis('off')
        except Exception as e:
            print(f"Error visualizing {img_path}: {e}")
    
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "sample_predictions.png")
    print(f"Saved visualization to {output_dir / 'sample_predictions.png'}")
    
    # Create a more comprehensive results file
    create_detailed_results(results, output_dir)

def create_detailed_results(results, output_dir):
    """Create a detailed CSV of results"""
    csv_path = output_dir / "inference_results.csv"
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['image', 'prediction', 'confidence', 'second_best', 'second_confidence', 
                     'third_best', 'third_confidence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow({
                'image': result['filename'],
                'prediction': result['prediction'],
                'confidence': f"{result['confidence']*100:.2f}%",
                'second_best': result['top3_classes'][1],
                'second_confidence': f"{result['top3_probs'][1]*100:.2f}%",
                'third_best': result['top3_classes'][2],
                'third_confidence': f"{result['top3_probs'][2]*100:.2f}%"
            })
    
    print(f"Saved detailed results to {csv_path}")

def print_summary_statistics(results):
    """Print summary statistics of the predictions"""
    if not results:
        print("No results to summarize.")
        return
        
    # Count predictions by class
    class_counts = {}
    for result in results:
        pred = result['prediction']
        if pred in class_counts:
            class_counts[pred] += 1
        else:
            class_counts[pred] = 1
    
    # Calculate average confidence
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    
    # Print summary
    print("\n===== Inference Summary =====")
    print(f"Total images processed: {len(results)}")
    print(f"Average confidence: {avg_confidence*100:.2f}%")
    print("\nPredictions by class:")
    
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(results)) * 100
        print(f"  {cls}: {count} images ({percentage:.1f}%)")

def main():
    """Main function to run inference"""
    try:
        print("Starting scene classification inference with transformer model...")
        results = run_inference_on_test_images()
        if results:
            print("\nInference completed successfully!")
        else:
            print("\nNo results generated. Please check the error messages above.")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 