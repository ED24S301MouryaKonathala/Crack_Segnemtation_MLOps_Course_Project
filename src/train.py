import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import albumentations as A
import mlflow
import mlflow.pytorch
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
import sys
from pathlib import Path
from matplotlib import pyplot as plt

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_params

# Setup logging
log_dir = Path(__file__).parent.parent / "logs" / "training"
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Load Params ---
try:
    params = load_params()
    train_params = params.get("train", {})
    dataset_params = params.get("dataset", {})
    num_epochs = train_params.get("epochs", 2)
    batch_size = train_params.get("batch_size", 8)
    learning_rate = train_params.get("learning_rate", 1e-4)
    img_size = train_params.get("image_size", 128)
    train_images_dir = dataset_params.get("train_images_dir", "data/train/images")
    train_masks_dir = dataset_params.get("train_masks_dir", "data/train/masks")
    test_images_dir = dataset_params.get("test_images_dir", "data/test/images")
    test_masks_dir = dataset_params.get("test_masks_dir", "data/test/masks")
    val_split = dataset_params.get("val_split", 0.1)
    logger.info("Parameters loaded from params.yaml")
except Exception as e:
    logger.error(f"Failed to load parameters: {e}")
    raise

# --- Attention Gate and Attention U-Net Implementation ---

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(img_ch, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)

        self.Up5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.Att5 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.Att4 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.Att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.Att2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.UpConv2 = ConvBlock(128, 64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool(x1))
        x3 = self.Conv3(self.Maxpool(x2))
        x4 = self.Conv4(self.Maxpool(x3))
        x5 = self.Conv5(self.Maxpool(x4))

        # Decoder + Attention
        d5 = self.Up5(x5)
        x4 = self.Att5(x4, d5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(x3, d4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(x2, d3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(x1, d2)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv_1x1(d2)
        return out

# --- Dataset ---
class CrackDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=128, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.img_size = img_size
        self.images = sorted(os.listdir(images_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx])

        image = Image.open(img_path).convert("RGB").resize((self.img_size, self.img_size))
        mask = Image.open(mask_path).convert("L").resize((self.img_size, self.img_size))

        image = np.array(image) / 255.0
        mask = np.array(mask) / 255.0
        mask = np.expand_dims(mask, axis=0)  # (1, H, W)

        if self.transform:
            augmented = self.transform(image=image, mask=mask[0])
            image = augmented['image']
            mask = augmented['mask'][None, :, :]

        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask

# --- Prepare train/val split from train folder ---
try:
    full_train_dataset = CrackDataset(train_images_dir, train_masks_dir, img_size=img_size, transform=None)
    indices = np.arange(len(full_train_dataset))
    train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=42, shuffle=True)
    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_train_dataset, val_idx)
    train_dataset.dataset.transform = A.Compose([A.HorizontalFlip(p=0.5)]) if dataset_params.get("augmentations", True) else None
    val_dataset.dataset.transform = None
    test_dataset = CrackDataset(test_images_dir, test_masks_dir, img_size=img_size, transform=None)
    logger.info(f"Train/Val/Test splits: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
except Exception as e:
    logger.error(f"Error preparing datasets: {e}")
    raise

# --- Dataloaders ---
try:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
except Exception as e:
    logger.error(f"Error creating dataloaders: {e}")
    raise

# --- Model, Loss, Optimizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttentionUNet(img_ch=3, output_ch=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def calc_dice_coefficient(pred, target):
    smooth = 1e-5
    pred = torch.sigmoid(pred) > 0.5
    pred = pred.float()
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

def calc_iou(pred, target):
    smooth = 1e-5
    pred = torch.sigmoid(pred) > 0.5
    pred = pred.float()
    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def calc_metrics(outputs, masks):
    preds = (torch.sigmoid(outputs) > 0.5).float()
    correct = (preds == masks).float().sum().item()
    total = torch.numel(masks)
    acc = (correct / total) * 100  # Convert to percentage
    return acc

def evaluate(loader, model, criterion, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    n_batches = 0
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, masks.squeeze(1))
            
            # Calculate metrics
            dice = calc_dice_coefficient(outputs, masks.squeeze(1))
            iou = calc_iou(outputs, masks.squeeze(1))
            
            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'dice': total_dice / n_batches,
        'iou': total_iou / n_batches
    }

def save_predictions(model, test_loader, device, run_name, img_size):
    model.eval()
    os.makedirs(f"predictions/{run_name}", exist_ok=True)
    count = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
            images = images.cpu().numpy()
            masks = masks.cpu().numpy()
            batch_size = images.shape[0]
            for i in range(batch_size):
                if count >= 5:
                    return
                fig, axs = plt.subplots(1, 3, figsize=(9, 3))
                img = np.transpose(images[i], (1, 2, 0))
                axs[0].imshow(img)
                axs[0].set_title("Image")
                axs[1].imshow(masks[i][0], cmap='gray')
                axs[1].set_title("Ground Truth")
                axs[2].imshow(preds[i][0], cmap='gray')
                axs[2].set_title("Prediction")
                for ax in axs:
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig(f"predictions/{run_name}/sample_{count+1}.png")
                plt.close(fig)
                count += 1
            if count >= 5:
                break

# --- MLflow Setup ---
try:
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("CrackSegmentation-AttentionUNet")
    run_name = f"training_run_{num_epochs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model": "Attention-UNet (PyTorch)",
            "img_size": img_size,
            "optimizer": optimizer.__class__.__name__,
            "loss_function": criterion.__class__.__name__
        })
        best_val_loss = float('inf')
        best_val_dice = 0  # Track best validation Dice score
        early_stopping_patience = 10
        patience_counter = 0
        best_model_path = None
        for epoch in range(num_epochs):
            try:
                model.train()
                epoch_loss = 0
                epoch_dice = 0
                epoch_iou = 0
                n_batches = 0
                for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                    images = images.to(device)
                    masks = masks.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    outputs = outputs.squeeze(1)
                    loss = criterion(outputs, masks.squeeze(1))
                    loss.backward()
                    optimizer.step()
                    
                    # Calculate training metrics
                    dice = calc_dice_coefficient(outputs, masks.squeeze(1))
                    iou = calc_iou(outputs, masks.squeeze(1))
                    
                    epoch_loss += loss.item()
                    epoch_dice += dice
                    epoch_iou += iou
                    n_batches += 1

                # Calculate average metrics
                avg_train_metrics = {
                    'loss': epoch_loss / n_batches,
                    'dice': epoch_dice / n_batches,
                    'iou': epoch_iou / n_batches
                }

                # Evaluate on validation set
                val_metrics = evaluate(val_loader, model, criterion, device)
                
                # Log metrics
                mlflow.log_metrics({
                    'train_loss': avg_train_metrics['loss'],
                    'train_dice': avg_train_metrics['dice'],
                    'train_iou': avg_train_metrics['iou'],
                    'val_loss': val_metrics['loss'],
                    'val_dice': val_metrics['dice'],
                    'val_iou': val_metrics['iou']
                }, step=epoch)

                logger.info(f"Epoch [{epoch+1}/{num_epochs}] "
                            f"Train Loss: {avg_train_metrics['loss']:.4f}, "
                            f"Train Dice: {avg_train_metrics['dice']:.4f}, "
                            f"Train IoU: {avg_train_metrics['iou']:.4f}, "
                            f"Val Loss: {val_metrics['loss']:.4f}, "
                            f"Val Dice: {val_metrics['dice']:.4f}, "
                            f"Val IoU: {val_metrics['iou']:.4f}")

                # Early stopping check based on validation Dice
                if val_metrics['dice'] > best_val_dice:
                    best_val_dice = val_metrics['dice']
                    patience_counter = 0
                    os.makedirs('models', exist_ok=True)
                    best_model_path = Path("models") / f"attention_unet_{run_name}.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_dice': best_val_dice,
                    }, best_model_path)
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break

            except Exception as e:
                logger.error(f"Error during epoch {epoch+1}: {str(e)}")
                raise

        # Final evaluation and model saving
        try:
            # Load best model for evaluation and MLflow logging
            if best_model_path and best_model_path.exists():
                checkpoint = torch.load(best_model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
            test_metrics = evaluate(test_loader, model, criterion, device)
            logger.info(f"Test Loss: {test_metrics['loss']:.4f}, "
                        f"Test Dice: {test_metrics['dice']:.4f}, "
                        f"Test IoU: {test_metrics['iou']:.4f}")
            
            mlflow.log_metrics({
                'test_loss': test_metrics['loss'],
                'test_dice': test_metrics['dice'],
                'test_iou': test_metrics['iou']
            })

            # Log best model to MLflow
            mlflow.pytorch.log_model(model, "attention_unet_model")

            # Save predictions
            save_predictions(model, test_loader, device, run_name, img_size)

        except Exception as e:
            logger.error(f"Error during final evaluation: {str(e)}")
            raise

except Exception as e:
    logger.error(f"Training failed: {str(e)}")
    raise
