import cv2
import numpy as np
import tensorflow as tf
import onnxmltools
# Charger les datasets


#██████╗░░█████╗░██████╗░░█████╗░███╗░░░███╗███████╗████████╗███████╗██████╗░░██████╗
#██╔══██╗██╔══██╗██╔══██╗██╔══██╗████╗░████║██╔════╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝
#██████╔╝███████║██████╔╝███████║██╔████╔██║█████╗░░░░░██║░░░█████╗░░██████╔╝╚█████╗░
#██╔═══╝░██╔══██║██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝░░░░░██║░░░██╔══╝░░██╔══██╗░╚═══██╗
#██║░░░░░██║░░██║██║░░██║██║░░██║██║░╚═╝░██║███████╗░░░██║░░░███████╗██║░░██║██████╔╝
#╚═╝░░░░░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░░░░╚═╝╚══════╝░░░╚═╝░░░╚══════╝╚═╝░░╚═╝╚═════╝░

DATASETNAME = "datasetV2"
DATASET_VAL_PATH = f"{DATASETNAME}/dataset/val"
DATASET_TEST_PATH = f"{DATASETNAME}/dataset/test"
DATASET_TRAIN_PATH = f"{DATASETNAME}/dataset/train"

NB_CLASSES_MODEL = 10


#░█████╗░░█████╗░██████╗░███████╗
#██╔══██╗██╔══██╗██╔══██╗██╔════╝
#██║░░╚═╝██║░░██║██║░░██║█████╗░░
#██║░░██╗██║░░██║██║░░██║██╔══╝░░
#╚█████╔╝╚█████╔╝██████╔╝███████╗
#░╚════╝░░╚════╝░╚═════╝░╚══════╝

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_TRAIN_PATH,
    image_size=(224, 224),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_VAL_PATH,
    image_size=(224, 224),
    batch_size=32
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_TEST_PATH,
    image_size=(224, 224),
    batch_size=32
)

# Normalisation des pixels
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))  # Ajout de la normalisation sur test_ds

# Charger le modèle pré-entraîné MobileNetV2
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False  # Geler les poids du modèle pré-entraîné

# Créer ton modèle
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(NB_CLASSES_MODEL, activation='softmax')  # 10 classes
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle
model.fit(train_ds, validation_data=val_ds, epochs=10)

# Évaluer le modèle sur l'ensemble de test
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Sauvegarder le modèle en .pb
model.export(f"models/{DATASETNAME}")
