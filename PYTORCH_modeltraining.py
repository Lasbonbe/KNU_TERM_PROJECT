import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

#********************************************************************************#
#████████╗██████╗░░█████╗░██╗███╗░░██╗██╗███╗░░██╗░██████╗░
#╚══██╔══╝██╔══██╗██╔══██╗██║████╗░██║██║████╗░██║██╔════╝░
#░░░██║░░░██████╔╝███████║██║██╔██╗██║██║██╔██╗██║██║░░██╗░
#░░░██║░░░██╔══██╗██╔══██║██║██║╚████║██║██║╚████║██║░░╚██╗
#░░░██║░░░██║░░██║██║░░██║██║██║░╚███║██║██║░╚███║╚██████╔╝
#░░░╚═╝░░░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝╚═╝░░╚══╝╚═╝╚═╝░░╚══╝░╚═════╝░
# USING PYTORCH
#
# This code trains a ResNet-18 model on the dataset and saves the model to a file.
# You can modify the data used by changing the DATASET variable.
DATASET = "datasetV3"
train_dir = f"{DATASET}/dataset/train"
val_dir = f"{DATASET}/dataset/val"
test_dir = f"{DATASET}/dataset/test"
# It makes 10 epochs of training and 1 test of validation
# It saves the model to a file named "{DATASET}_PYTORCH.pt"
#********************************************************************************#

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# TRAIN VAL TERT DATASETS
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# DATALOADERS
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Charger un modèle ResNet pré-entraîné
model = resnet18(pretrained=True)

# Modifier la couche de sortie pour s'adapter au nombre de classes dans votre dataset
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

criterion = nn.CrossEntropyLoss()  # Pour un problème de classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Boucle d'entraînement
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Validation Accuracy: {100 * correct / total:.2f}%")

torch.save(model, f"models/{DATASET}_PYTORCH.pt")
