import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from anomalib.models import EfficientAD
import mlflow
import mlflow.pytorch
import yaml

# Load parameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)

data_dir = "data/"
batch_size = params["batch_size"]
epochs = params["epochs"]
learning_rate = params["learning_rate"]

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EfficientAD().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

mlflow.start_run()

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for images, _ in train_loader:
        images = images.to(device)
        labels = torch.ones(images.size(0)).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(torch.sigmoid(outputs.squeeze()), labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    mlflow.log_metric("loss", avg_loss)

mlflow.pytorch.log_model(model, "efficientad_model")
torch.save(model, "models/efficientad_model.pth")

mlflow.end_run()
