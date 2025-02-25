from huggingface_hub import HfApi, create_repo
from transformers import ViTImageProcessor, ViTConfig
from dotenv import load_dotenv
import json
import os
import shutil

# Load environment variables
load_dotenv()

# Hugging Face Hub settings
REPO_NAME = "SebLogsdon/EveryScene"  # Change this
TOKEN = os.getenv('HF_TOKEN')

# Model paths, change for your model
MODEL_DIR = "./models_3/best_model"
REQUIRED_FILES = [
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "classifier_head.pth"
]

if not TOKEN:
    raise ValueError("No Hugging Face token found in .env file. Please add HUGGINGFACE_TOKEN=your_token_here to your .env file.")

# Model card content remains the same
model_card = """
---
language: en
tags:
... (rest of the model card content remains the same)
"""

def verify_model_files():
    """Verify all required model files exist"""
    if not os.path.exists(MODEL_DIR):
        raise ValueError(f"Model directory not found at {MODEL_DIR}")
    
    missing_files = []
    for file in REQUIRED_FILES:
        file_path = os.path.join(MODEL_DIR, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        raise ValueError(f"Missing required model files: {', '.join(missing_files)}")
    
    print("All required model files found!")
    return True

def upload_model_to_hub():
    # Verify model files first
    verify_model_files()
    
    # Create API client
    api = HfApi()
    
    # Create repository
    try:
        create_repo(
            repo_id=REPO_NAME,
            token=TOKEN,
            private=False,
            exist_ok=True
        )
        print(f"Repository {REPO_NAME} created/accessed successfully")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    # Upload files
    try:
        # Upload model files
        for file in REQUIRED_FILES:
            file_path = os.path.join(MODEL_DIR, file)
            print(f"Uploading {file}...")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file,
                repo_id=REPO_NAME,
                token=TOKEN
            )
        
        # Create and upload README
        readme_path = os.path.join(MODEL_DIR, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(model_card)
        
        print("Uploading README.md...")
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=REPO_NAME,
            token=TOKEN
        )
        
        print("\nUpload completed successfully!")
        print(f"Model available at: https://huggingface.co/{REPO_NAME}")
        
    except Exception as e:
        print(f"Error during upload: {e}")
        raise

if __name__ == "__main__":
    try:
        upload_model_to_hub()
    except Exception as e:
        print(f"Failed to upload model: {e}")
