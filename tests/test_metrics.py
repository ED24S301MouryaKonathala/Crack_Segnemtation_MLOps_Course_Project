import torch
import numpy as np
from src.train import calc_dice_coefficient, calc_iou
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dice_coefficient():
    try:
        pred = torch.tensor([[1, 1], [0, 0]], dtype=torch.float32)
        target = torch.tensor([[1, 1], [0, 0]], dtype=torch.float32)
        dice = calc_dice_coefficient(pred, target)
        if abs(dice - 1.0) > 1e-6:
            logger.error(f"Incorrect Dice score: {dice}")
            return False
        logger.info("Dice coefficient test passed")
        return True
    except Exception as e:
        logger.error(f"Dice coefficient test failed: {e}")
        return False

def test_iou():
    try:
        pred = torch.tensor([[1, 1], [0, 0]], dtype=torch.float32)
        target = torch.tensor([[1, 1], [0, 0]], dtype=torch.float32)
        iou = calc_iou(pred, target)
        if abs(iou - 1.0) > 1e-6:
            logger.error(f"Incorrect IoU score: {iou}")
            return False
        logger.info("IoU test passed")
        return True
    except Exception as e:
        logger.error(f"IoU test failed: {e}")
        return False

if __name__ == "__main__":
    test_dice_coefficient()
    test_iou()
