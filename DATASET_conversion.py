import os
from PIL import Image
import shutil
from torchvision import transforms

#********************************************************************************#
#░█████╗░░█████╗░███╗░░██╗██╗░░░██╗███████╗██████╗░░██████╗██╗░█████╗░███╗░░██╗
#██╔══██╗██╔══██╗████╗░██║██║░░░██║██╔════╝██╔══██╗██╔════╝██║██╔══██╗████╗░██║
#██║░░╚═╝██║░░██║██╔██╗██║╚██╗░██╔╝█████╗░░██████╔╝╚█████╗░██║██║░░██║██╔██╗██║
#██║░░██╗██║░░██║██║╚████║░╚████╔╝░██╔══╝░░██╔══██╗░╚═══██╗██║██║░░██║██║╚████║
#╚█████╔╝╚█████╔╝██║░╚███║░░╚██╔╝░░███████╗██║░░██║██████╔╝██║╚█████╔╝██║░╚███║
#░╚════╝░░╚════╝░╚═╝░░╚══╝░░░╚═╝░░░╚══════╝╚═╝░░╚═╝╚═════╝░╚═╝░╚════╝░╚═╝░░╚══╝
# This code converts YOLO-formatted labels to PyTorch-formatted labels
#(•_•)
#( •_•)>⌐■-■
#(⌐■_■)
# It's that easy!
#********************************************************************************#

source_image_dir = 'Dataset military equipment/images_train'
source_label_dir = 'Dataset military equipment/labels_train'
target_image_dir = 'datasetV1/dataset/train/images'
target_label_dir = 'datasetV1/dataset/train/labels'

transform = transforms.Compose([
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalization
])
os.makedirs(target_image_dir, exist_ok=True)
os.makedirs(target_label_dir, exist_ok=True)

for image_file in os.listdir(source_image_dir):
    img_path = os.path.join(source_image_dir, image_file)
    image = Image.open(img_path).convert('RGB')

    image_tensor = transform(image)

    image = transforms.ToPILImage()(image_tensor)

    target_img_path = os.path.join(target_image_dir, image_file)
    image.save(target_img_path)

    label_file = image_file.replace('.jpg', '.txt')
    source_label_path = os.path.join(source_label_dir, label_file)
    target_label_path = os.path.join(target_label_dir, label_file)

    if os.path.exists(source_label_path):
        shutil.copy(source_label_path, target_label_path)
    else:
        open(target_label_path, 'w').close()

print('Conversion complete.')