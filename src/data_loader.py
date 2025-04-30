import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_data_loader(data_path, batch_size):
    try:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        dataset = ImageFolder(root=data_path, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        logger.info(f"Loaded data from {data_path} with {len(dataset)} samples, batch size {batch_size}")
        return loader
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
