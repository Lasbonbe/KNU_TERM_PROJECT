import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim


# --- Dataset Loader ---
class MilitaryEquipmentDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace('.jpg', '.txt'))

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Load labels (YOLO format: [class_id, x_center, y_center, width, height])
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = [line.strip().split() for line in f.readlines()]
                labels = [[float(value) for value in label] for label in labels]
                labels = torch.tensor(labels, dtype=torch.float32)
        else:
            labels = torch.tensor([])

        if self.transform:
            image = self.transform(image)

        return image, labels


# --- Data Transformation ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- Dataset Paths ---
train_dataset = MilitaryEquipmentDataset(
    image_dir="Dataset military equipment/images_train",
    label_dir="Dataset military equipment/labels_train",
    transform=transform,
)

val_dataset = MilitaryEquipmentDataset(
    image_dir="Dataset military equipment/images_val",
    label_dir="Dataset military equipment/labels_val",
    transform=transform,
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# --- Model Definition ---
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)  # Assuming 10 classes (adjust as needed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# --- Loss and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# --- Training Loop ---
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = dataloaders['train']
            else:
                model.eval()
                loader = dataloaders['val']

            running_loss = 0.0

            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(loader.dataset)
            print(f"{phase} Loss: {epoch_loss:.4f}")

    print("Training complete")


dataloaders = {
    'train': train_loader,
    'val': val_loader,
}

train_model(model, dataloaders, criterion, optimizer)

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

torch.save(model, f"models/TEST_PYTORCH.pt")