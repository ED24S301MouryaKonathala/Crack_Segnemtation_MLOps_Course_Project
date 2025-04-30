import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import mlflow
import mlflow.pytorch
from PIL import Image
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import yaml

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from backend.app.utils import load_model, AttentionUNet

# Setup logging
log_dir = Path(__file__).parent.parent / "logs" / "retrain"
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "retrain.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_retrain_params():
    try:
        params_path = Path(__file__).parent / "retrain_params.yaml"
        with open(params_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load retrain parameters: {e}")
        raise

class RetrainDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=128):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        
        # Get all image files
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')) + list(self.images_dir.glob('*.png')))
        
        # Check for missing masks
        self.valid_pairs = []
        self.missing_masks = []
        
        for img_path in self.image_files:
            mask_path = self.masks_dir / img_path.name
            if mask_path.exists():
                self.valid_pairs.append((img_path, mask_path))
            else:
                self.missing_masks.append(img_path.name)
                
        if self.missing_masks:
            logger.warning(f"Missing masks for images: {', '.join(self.missing_masks)}")
            
    def __len__(self):
        return len(self.valid_pairs)
        
    def __getitem__(self, idx):
        img_path, mask_path = self.valid_pairs[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # Resize
        image = image.resize((self.img_size, self.img_size))
        mask = mask.resize((self.img_size, self.img_size))
        
        # Convert to numpy arrays and normalize
        image = np.array(image) / 255.0
        mask = np.array(mask) / 255.0
        
        # Convert to tensors
        image = torch.FloatTensor(image.transpose(2, 0, 1))
        mask = torch.FloatTensor(mask).unsqueeze(0)
        
        return image, mask

def retrain_model():
    try:
        # Load retrain params
        params = load_retrain_params()
        train_params = params.get("train", {})
        
        # Setup paths relative to model_retrain directory
        base_dir = Path(__file__).parent.parent
        retrain_data_dir = Path(__file__).parent / "model_retrain_data"
        images_dir = retrain_data_dir / "images"
        masks_dir = retrain_data_dir / "masks"
        
        # Check directories exist
        if not all(path.exists() for path in [retrain_data_dir, images_dir, masks_dir]):
            raise FileNotFoundError("Required directories not found. Ensure model_retrain_data/images and masks exist.")
            
        # Load latest model
        model = load_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Setup training parameters
        num_epochs = train_params.get("epochs", 10)
        batch_size = train_params.get("batch_size", 4)
        learning_rate = train_params.get("learning_rate", 1e-4)
        img_size = train_params.get("image_size", 128)
        
        # Create dataset and dataloader
        dataset = RetrainDataset(images_dir, masks_dir, img_size)
        if not dataset.valid_pairs:
            raise ValueError("No valid image-mask pairs found for retraining")
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup training
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # MLflow logging
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("CrackSegmentation-Retraining")
        run_name = f"retrain_run_{num_epochs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "img_size": img_size,
                "num_images": len(dataset)
            })
            
            # Training loop
            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0
                
                for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                mlflow.log_metric("loss", avg_loss, step=epoch)
                logger.info(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
            
            # Save retrained model
            save_path = base_dir / "models" / f"attention_unet_retrain_run_{num_epochs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': num_epochs,
            }, save_path)
            
            logger.info(f"Model retrained and saved to {save_path}")
            mlflow.pytorch.log_model(model, "retrained_model")
            
    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}")
        raise

if __name__ == "__main__":
    retrain_model()