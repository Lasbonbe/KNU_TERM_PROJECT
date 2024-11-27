import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import cv2
import numpy as np

# LOAD DATASET AND MODEL
DATASET = "datasetV3"
MODEL = "datasetV3_TENSORFLOW"

# LOAD MODEL INTO CV2 DNN
model = cv2.dnn.readNetFromONNX(f"ONNX_models/{MODEL}.onnx")

# LOAD IMAGE
IMAGE_PATH = "images_for_test/vehicule.jpg"
IMAGE_PATH = "assets/misile_test_1.jpg"
image = cv2.imread(IMAGE_PATH)
# RESIZE IMAGE
image = cv2.resize(image, (1000, 500))

# AFFICHAGE DE L'IMAGE
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# PREPROCESSING

# Resize the image but keep the aspect ratio
def resize_image(image, width=None, height=None):
    h, w = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        ratio = height / h
        width = int(w * ratio)
    elif height is None:
        ratio = width / w
        height = int(h * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


# DETECT VEHICLES
blob = cv2.dnn.blobFromImage(image, 1/255, (224, 224), (0, 0, 0), swapRB=True, crop=False)
model.setInput(blob)
output = model.forward()

# GET THE PREDICTION
_, predicted = torch.max(torch.tensor(output), 1)
print(predicted.item())

# 1 = tank
# 2 = SPAA
# 0 = Military vehicle

# DISPLAY THE PREDICTION
if predicted.item() == 1:
    print("Tank")
elif predicted.item() == 2:
    print("SPAA")
else:
    print("Military vehicle")

# LOAD CURSOR
cursor = cv2.imread("assets/MISSILE_CURSOR.png", cv2.IMREAD_UNCHANGED)

# PUT CURSOR IN THE IMAGE CENTER NO RESIZE
h, w = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 0, 1)
rotated_cursor = cv2.warpAffine(cursor, M, (w, h))













