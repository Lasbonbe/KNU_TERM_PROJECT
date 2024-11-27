import os
from PIL import Image
import shutil
import torch
from torchvision import transforms

# Define the transformations for the data (without resizing)
transform = transforms.Compose([
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalization
])

# Paths to the data
source_image_dir = 'Dataset military equipment/images_train'
source_label_dir = 'Dataset military equipment/labels_train'
target_image_dir = 'datasetV1/dataset/train/images'
target_label_dir = 'datasetV1/dataset/train/labels'

# Create target directories if they do not exist
os.makedirs(target_image_dir, exist_ok=True)
os.makedirs(target_label_dir, exist_ok=True)

# Convert and copy images and labels
for image_file in os.listdir(source_image_dir):
    # Load the image
    img_path = os.path.join(source_image_dir, image_file)
    image = Image.open(img_path).convert('RGB')

    # Apply the transformations
    image_tensor = transform(image)

    # Convert the tensor back to a PIL image
    image = transforms.ToPILImage()(image_tensor)

    # Save the transformed image
    target_img_path = os.path.join(target_image_dir, image_file)
    image.save(target_img_path)

    # Copy the corresponding label
    label_file = image_file.replace('.jpg', '.txt')
    source_label_path = os.path.join(source_label_dir, label_file)
    target_label_path = os.path.join(target_label_dir, label_file)

    if os.path.exists(source_label_path):
        shutil.copy(source_label_path, target_label_path)
    else:
        # Create an empty label file if the label does not exist
        open(target_label_path, 'w').close()

print('Conversion complete.')