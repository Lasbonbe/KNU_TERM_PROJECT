import cv2

# Paths to .pb and .pbtxt files
model_path = "models/datasetV2/saved_model.pb"


# Load the model
net = cv2.dnn.readNetFromTensorflow(model_path)