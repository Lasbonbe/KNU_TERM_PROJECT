import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from PIL import Image

# Charger une image et appliquer les transformations
img = Image.open("images_for_test/tank2.jpeg")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
img = transform(img).unsqueeze(0)

# Charger le modèle
MODEL = "modelV2_PYTORCH"
MODEL = "datasetV1_PYTORCH"
model = torch.load(f"models/{MODEL}.pt")

# Faire une prédiction
model.eval()
output = model(img)
_, predicted = torch.max(output, 1)

print(predicted.item())