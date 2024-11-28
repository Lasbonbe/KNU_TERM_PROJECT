import torch
import cv2
from TESTING import resize_coordinates, add_cursor_to_image, hex_to_rgb, preview
import numpy as np

src_width, src_height = 5120, 2880 # RESOLUTION OF TEST IMAGES
dest_width, dest_height = 1720, 720 # RESOLUTION OF DISPLAYED IMAGES

# LOAD DATASET AND MODEL
DATASET = "datasetV3"
MODEL = "datasetV3_TENSORFLOW"
model = cv2.dnn.readNetFromONNX(f"ONNX_models/{MODEL}.onnx")
cursor = cv2.imread("assets/MISSILE_CURSOR.png", cv2.IMREAD_UNCHANGED)


def get_bounding_box(prediction_output):
    detection_threshold = 0.5
    boxes = []

    for detection in prediction_output.squeeze():
        confidence = detection[2]
        if confidence > detection_threshold:
            (x_center, y_center, w, h) = (detection[3], detection[4], detection[5], detection[6])
            x1 = int((x_center - w / 2) * dest_width)
            y1 = int((y_center - h / 2) * dest_height)
            x2 = int((x_center + w / 2) * dest_width)
            y2 = int((y_center + h / 2) * dest_height)
            boxes.append((detection[1], confidence, (x1, y1, x2, y2)))

    return boxes


# LOAD IMAGE
IMAGE_PATH = "images_for_test/tank.jpg"
image = cv2.imread(IMAGE_PATH)
image = cv2.resize(image, (1720, 720))

# DETECT VEHICLES (first frame detection)
blob = cv2.dnn.blobFromImage(image, 1 / 255, (224, 224), (0, 0, 0), swapRB=True, crop=False)
model.setInput(blob)
preview(IMAGE_PATH)
output = model.forward()
output = output.squeeze()
boxes = get_bounding_box(output)

if boxes:
    class_id, confidence, (x1, y1, x2, y2) = boxes[0]
    print(f"Detected class: {class_id}")
    centertank = ((x1 + x2) // 2, (y1 + y2) // 2)
else:
    print("No object detected.")
    exit(0)

image_height, image_width = image.shape[:2]
cursor_height, cursor_width = cursor.shape[:2]
zoom_factor = 1.0
max_zoom = 3


blob = cv2.dnn.blobFromImage(image, 1 / 255, (224, 224), (0, 0, 0), swapRB=True, crop=False)
model.setInput(blob)
output = model.forward()
output = output.squeeze()
boxes = get_bounding_box(output)

if boxes:
    class_id, confidence, (x1, y1, x2, y2) = boxes[0]
    centertank = ((x1 + x2) // 2, (y1 + y2) // 2)
else:
    print("No object detected.")
    exit(0)

image_height, image_width = image.shape[:2]
zoom_factor = 1.0
max_zoom = 3

while zoom_factor < max_zoom:
    crop_height = int(image_height / zoom_factor)
    crop_width = int(image_width / zoom_factor)
    start_y = max(0, centertank[1] - crop_height // 2)
    start_x = max(0, centertank[0] - crop_width // 2)
    end_y = min(image_height, start_y + crop_height)
    end_x = min(image_width, start_x + crop_width)
    resize_coordinates(start_x, start_y, src_width, src_height, dest_width, dest_height)

    if start_y < end_y and start_x < end_x:
        cropped_image = image[start_y:end_y, start_x:end_x]

        if cropped_image.size != 0:
            zoomed_image = cv2.resize(cropped_image, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
            displayed_image = add_cursor_to_image(zoomed_image.copy(), cursor)

            adj_x1 = max(0, x1 - start_x)
            adj_y1 = max(0, y1 - start_y)
            adj_x2 = min(crop_width, x2 - start_x)
            adj_y2 = min(crop_height, y2 - start_y)

            if class_id == 2:
                cv2.rectangle(displayed_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(displayed_image, "Tank", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            elif class_id == 1:
                cv2.rectangle(displayed_image, (x1, y1), (x2, y2), (222, 12, 255), 2)
                cv2.putText(displayed_image, "SPAA", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (222, 12, 255), 2)
            else:
                cv2.rectangle(displayed_image, (x1, y1), (x2, y2), (15, 172, 255), 2)
                cv2.putText(displayed_image, "Military vehicle", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (15, 172, 255), 2)

            cv2.imshow("Zooming Image", displayed_image)
            key = cv2.waitKey(2) & 0xFF
            if key == 27:  # Exit on pressing Esc key
                break
            zoom_factor += 0.005

cv2.destroyAllWindows()
exit(0)
