import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_params(path="params.yaml"):
    try:
        with open(path) as file:
            params = yaml.safe_load(file)
        logger.info(f"Loaded parameters from {path}")
        return params
    except Exception as e:
        logger.error(f"Failed to load parameters from {path}: {e}")
        raise
