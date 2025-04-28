from src.data_loader import get_data_loader
from src.utils import load_params

def test_data_loader():
    params = load_params()
    loader = get_data_loader("data/", batch_size=params["batch_size"])
    batch = next(iter(loader))
    assert batch[0].shape[0] == params["batch_size"]

def test_params():
    params = load_params()
    assert "batch_size" in params
    assert "epochs" in params
    assert "learning_rate" in params
