from huggingface_hub import hf_hub_download
import os
from pathlib import Path
import shutil
from tqdm import tqdm

# Hugging Face Hub settings
REPO_ID = "SebLogsdon/EveryScene"
MODEL_FILES = [
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "classifier_head.pth"
]

def download_model():
    """Download the scene classifier model from Hugging Face Hub"""
    print(f"Downloading scene classifier model from {REPO_ID}...")
    
    # Create model directory
    model_dir = Path("./tmp/scene_models/transformer")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Download each file
    for file in tqdm(MODEL_FILES, desc="Downloading model files"):
        try:
            # Download file from Hugging Face Hub
            file_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=file,
                cache_dir="./tmp/hf_cache"
            )
            
            # Copy to model directory
            dest_path = model_dir / file
            shutil.copy(file_path, dest_path)
            print(f"Downloaded {file} to {dest_path}")
            
        except Exception as e:
            print(f"Error downloading {file}: {e}")
            raise
    
    print(f"\nModel downloaded successfully to {model_dir}")
    return model_dir

def main():
    """Main function to download the model"""
    try:
        model_dir = download_model()
        print(f"Scene classifier transformer model is ready at {model_dir}")
        return True
    except Exception as e:
        print(f"Failed to download model: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
