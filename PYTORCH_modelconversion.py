import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

MODEL="datasetV1_PYTORCH"

model = torch.load(f"models/{MODEL}.pt")
model.eval()

#SAVE IN ONNX FORMAT FOR CV2.DNN
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, f"ONNX_models/{MODEL}.onnx")
