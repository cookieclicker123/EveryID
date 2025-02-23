from typing import List, Tuple
import torch
import torch.nn as nn
from transformers import ViTModel, ViTFeatureExtractor
import torchvision.transforms as T
from tqdm.auto import tqdm
import numpy as np
import os
from PIL import Image
import shutil
import time

# Constants
IMG_SIZE = 224
EMBEDDING_DIM = 512
DROPOUT_PROB = 0.3
MODEL_PATH = "./tmp/person_models/person_reid_model_vit_msmt17_final.pth"
FOLDER_PATH = "./tmp/datasets/people"
OUTPUT_DIR = "./tmp/reid_results/people_reid_msmt17"

def init_model(model_path: str) -> Tuple[nn.Module, ViTFeatureExtractor]:
    """Initialize the model and feature extractor."""
    class PersonReIDTransformer(nn.Module):
        def __init__(self, embedding_dim=512, dropout_prob=0.3):
            super(PersonReIDTransformer, self).__init__()
            self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            self.embedding = nn.Sequential(
                nn.Linear(self.backbone.config.hidden_size, embedding_dim),
                nn.Dropout(p=dropout_prob)
            )

        def forward(self, x):
            outputs = self.backbone(x).last_hidden_state[:, 0]  # Extract [CLS] token
            embeddings = self.embedding(outputs)
            return embeddings
    
    # Initialize model with same architecture as training
    model = PersonReIDTransformer(embedding_dim=EMBEDDING_DIM, dropout_prob=DROPOUT_PROB)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    return model, feature_extractor

def load_images_from_folder(folder: str) -> Tuple[List[Image.Image], List[str]]:
    """Load all jpg images from a folder."""
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('RGB')
            images.append(img)
            filenames.append(filename)
    return images, filenames

def get_transformation_chain() -> T.Compose:
    """Create the image transformation pipeline."""
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def extract_embeddings(model: nn.Module, images: List[Image.Image], batch_size: int = 32) -> torch.Tensor:
    """Extract embeddings from images in batches."""
    device = next(model.parameters()).device
    transform = get_transformation_chain()
    all_embeddings = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        image_batch = torch.stack([transform(image) for image in batch])
        with torch.no_grad():
            embeddings = model(image_batch.to(device))
        all_embeddings.append(embeddings.cpu())
    
    return torch.cat(all_embeddings, dim=0)

def compute_similarity(emb_one: torch.Tensor, emb_two: torch.Tensor) -> List[float]:
    """Compute cosine similarity between embeddings."""
    return torch.nn.functional.cosine_similarity(emb_one, emb_two).numpy().tolist()

def get_similar_indices(query_embedding: torch.Tensor, all_embeddings: torch.Tensor, top_k: int = 20) -> List[int]:
    """Get indices of most similar embeddings."""
    sim_scores = compute_similarity(all_embeddings, query_embedding)
    return np.argsort(sim_scores)[::-1][:top_k].tolist()

def setup_output_directory(output_dir: str) -> None:
    """Set up the output directory structure."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Create reid_results folder if it doesn't exist
    reid_results_dir = "./tmp/reid_results"
    os.makedirs(reid_results_dir, exist_ok=True)

def save_similar_images(
    images: List[Image.Image],
    filenames: List[str],
    all_embeddings: torch.Tensor,
    output_dir: str
) -> None:
    """Process and save similar images for each query image."""
    for i, (image, filename) in enumerate(tqdm(zip(images, filenames), total=len(images))):
        subdir = os.path.join(output_dir, f"{i:04d}_{filename.split('.')[0]}")
        os.makedirs(subdir)
        
        # Save original image
        image.save(os.path.join(subdir, filename))
        
        # Find and save similar images
        query_embedding = all_embeddings[i].unsqueeze(0)
        sim_indices = get_similar_indices(query_embedding, all_embeddings)
        
        for j, idx in enumerate(sim_indices):
            if idx != i:
                similar_image = images[idx]
                similar_filename = filenames[idx]
                similar_image.save(os.path.join(subdir, f"{j:02d}_{similar_filename}"))

def main() -> None:
    """Main execution function."""
    start_time = time.time()
    
    # Initialize model and load images
    model, _ = init_model(MODEL_PATH)
    images, filenames = load_images_from_folder(FOLDER_PATH)
    
    # Setup device and extract embeddings
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    all_embeddings = extract_embeddings(model, images)
    
    # Process and save results
    setup_output_directory(OUTPUT_DIR)
    save_similar_images(images, filenames, all_embeddings, OUTPUT_DIR)
    
    print(f"Re-ID results saved in {OUTPUT_DIR}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
