from huggingface_hub import  whoami, hf_hub_download
import os

def check_hf_cli_login():
    try:
        user_info = whoami()
        return True, user_info['name']
    except Exception:
        return False, None

def download_model_from_hf():
    # Check if user is logged in via CLI
    is_logged_in, username = check_hf_cli_login()
    
    if not is_logged_in:
        print("Error: Not logged in to Hugging Face CLI")
        print("Please run: huggingface-cli login")
        return
    
    print(f"Authenticated as: {username}")
    
    try:
        # Create the scene_models directory if it doesn't exist
        download_dir = "./tmp/scene_models"
        os.makedirs(download_dir, exist_ok=True)
        
        # Define the repo name and model file
        repo_id = f"{username}/EveryScene"
        model_file = "scene_reid_model_places205_reduced.keras"
        
        # Download the model
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_file,
            local_dir=download_dir,
            local_dir_use_symlinks=False  # Get actual file, not symlink
        )
        
        print(f"\nSuccess! Model downloaded to: {downloaded_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nPlease check that:")
        print("1. You're properly logged in (huggingface-cli login)")
        print("2. The repository exists")
        print("3. You have read permissions for the repository")

if __name__ == "__main__":
    download_model_from_hf()