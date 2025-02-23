from huggingface_hub import HfApi, whoami

def check_hf_cli_login():
    try:
        user_info = whoami()
        return True, user_info['name']
    except Exception:
        return False, None

def upload_model_to_hf():
    # Check if user is logged in via CLI
    is_logged_in, username = check_hf_cli_login()
    
    if not is_logged_in:
        print("Error: Not logged in to Hugging Face CLI")
        print("Please run: huggingface-cli login")
        return
    
    print(f"Authenticated as: {username}")
    
    try:
        # Initialize the Hugging Face API
        api = HfApi()
        
        # Define the repo name
        repo_id = f"{username}/EveryScene"
        
        # Create the repo if it doesn't exist
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=False,
            exist_ok=True
        )

        # Upload the model file
        model_path = "./tmp/scene_models/scene_reid_model_places205_reduced.keras"
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="scene_reid_model_places205_reduced.keras",
            repo_id=repo_id
        )
        
        print(f"\nSuccess! Model uploaded to: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nPlease check that:")
        print("1. You're properly logged in (huggingface-cli login)")
        print("2. Your token has write permissions")
        print("3. The model file exists at the specified path")

if __name__ == "__main__":
    upload_model_to_hf()