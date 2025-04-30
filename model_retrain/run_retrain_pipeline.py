import os
import sys
import logging
import subprocess
from pathlib import Path
import yaml

# Setup logging
log_dir = Path(__file__).parent.parent / "logs" / "pipeline"
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
RETRAIN_THRESHOLD = 10  # Minimum number of image-mask pairs needed

def run_dvc_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        logger.info(f"DVC command succeeded: {command}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"DVC command failed: {command}\nError: {e.stderr}")
        return False

def check_data_threshold():
    try:
        retrain_data_dir = Path(__file__).parent / "model_retrain_data"
        images_dir = retrain_data_dir / "images"
        masks_dir = retrain_data_dir / "masks"

        if not all(path.exists() for path in [images_dir, masks_dir]):
            logger.error("Required directories not found")
            return False

        image_files = set(f.name for f in images_dir.glob('*.[jp][pn][g]'))
        mask_files = set(f.name for f in masks_dir.glob('*.[jp][pn][g]'))
        valid_pairs = image_files.intersection(mask_files)

        logger.info(f"Found {len(valid_pairs)} valid image-mask pairs")
        return len(valid_pairs) >= RETRAIN_THRESHOLD

    except Exception as e:
        logger.error(f"Error checking data threshold: {e}")
        return False

def track_with_dvc():
    try:
        # Add data to DVC
        if not run_dvc_command("dvc add model_retrain_data"):
            return False
            
        # Commit changes
        if not run_dvc_command("git add model_retrain_data.dvc"):
            return False
            
        if not run_dvc_command('git commit -m "Add new training data"'):
            return False
            
        # Push to remote storage
        if not run_dvc_command("dvc push"):
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error tracking with DVC: {e}")
        return False

def main():
    try:
        if not check_data_threshold():
            logger.info(f"Not enough data for retraining (threshold: {RETRAIN_THRESHOLD})")
            return

        logger.info("Starting retraining pipeline")
        
        # Track data with DVC
        if not track_with_dvc():
            logger.error("Failed to track data with DVC")
            return

        # Import here to avoid circular imports
        from model_retrain import retrain_model
        
        # Run retraining
        retrain_model()
        
        # Track new model with DVC
        if not run_dvc_command("dvc add models/*.pth"):
            logger.error("Failed to track new model with DVC")
            return
            
        logger.info("Retraining pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
