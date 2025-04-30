from src.data_loader import get_data_loader
from src.utils import load_params
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_loader():
    try:
        params = load_params()
        loader = get_data_loader("data/", batch_size=params["batch_size"])
        batch = next(iter(loader))
        if batch[0].shape[0] != params["batch_size"]:
            logger.error(f"Incorrect batch size: {batch[0].shape[0]}")
            return False
        logger.info("Data loader test passed")
        return True
    except Exception as e:
        logger.error(f"Data loader test failed: {e}")
        return False

def test_params():
    try:
        params = load_params()
        required = ["batch_size", "epochs", "learning_rate"]
        missing = [p for p in required if p not in params]
        if missing:
            logger.error(f"Missing parameters: {missing}")
            return False
        logger.info("Parameters test passed")
        return True
    except Exception as e:
        logger.error(f"Parameters test failed: {e}")
        return False

if __name__ == "__main__":
    test_data_loader()
    test_params()
