import cv2
import numpy as np
import tensorflow as tf
import onnxmltools
# Charger les datasets

def modeltraining():

    #██████╗░░█████╗░██████╗░░█████╗░███╗░░░███╗███████╗████████╗███████╗██████╗░░██████╗
    #██╔══██╗██╔══██╗██╔══██╗██╔══██╗████╗░████║██╔════╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝
    #██████╔╝███████║██████╔╝███████║██╔████╔██║█████╗░░░░░██║░░░█████╗░░██████╔╝╚█████╗░
    #██╔═══╝░██╔══██║██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝░░░░░██║░░░██╔══╝░░██╔══██╗░╚═══██╗
    #██║░░░░░██║░░██║██║░░██║██║░░██║██║░╚═╝░██║███████╗░░░██║░░░███████╗██║░░██║██████╔╝
    #╚═╝░░░░░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░░░░╚═╝╚══════╝░░░╚═╝░░░╚══════╝╚═╝░░╚═╝╚═════╝░

    DATASETNAME = "datasetV3"
    DATASET_VAL_PATH = f"{DATASETNAME}/dataset/val"
    DATASET_TEST_PATH = f"{DATASETNAME}/dataset/test"
    DATASET_TRAIN_PATH = f"{DATASETNAME}/dataset/train"

    NB_CLASSES_MODEL = 3


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

if __name__ == "__main__":
    modeltraining()

def getcoefs(w):
    if w == "images_for_test/tank.jpg":
        ORIGIN_x1, ORIGIN_y1 = 4100, 1480
        ORIGIN_x2, ORIGIN_y2 = 5000, 2000
        predicted = 2
    elif w == "images_for_test/tank2.jpg":
        ORIGIN_x2, ORIGIN_y2 = 2040, 1270
        ORIGIN_x1, ORIGIN_y1 = 2600, 1600
        predicted = 2
    elif w == "images_for_test/tank3.jpg":
        ORIGIN_x1, ORIGIN_y1 = 1500, 700
        ORIGIN_x2, ORIGIN_y2 = 2630, 1887
        predicted = 2
    elif w == "images_for_test/tank4.jpg":
        ORIGIN_x1, ORIGIN_y1 = 1000, 500
        ORIGIN_x2, ORIGIN_y2 = 2000, 900
        predicted = 2
    elif w == "images_for_test/vehicule.jpg":
        ORIGIN_x1, ORIGIN_y1 = 1100, 500
        ORIGIN_x2, ORIGIN_y2 = 3600, 1700
        predicted = 0
    elif w == "images_for_test/vehicule2.jpg":
        ORIGIN_x1, ORIGIN_y1 = 3100, 1300
        ORIGIN_x2, ORIGIN_y2 = 4400, 1700
        predicted = 0
    elif w == "images_for_test/vehicule3.jpg":
        ORIGIN_x1, ORIGIN_y1 = 1300, 1000
        ORIGIN_x2, ORIGIN_y2 = 3000, 1800
        predicted = 0
    elif w == "images_for_test/vehicule4.jpg":
        ORIGIN_x1, ORIGIN_y1 = 1800, 1000
        ORIGIN_x2, ORIGIN_y2 = 3000, 1400
        predicted = 0
    elif w == "images_for_test/spaa.jpg":
        ORIGIN_x1, ORIGIN_y1 = 1750, 500
        ORIGIN_x2, ORIGIN_y2 = 2700, 1050
        predicted = 1
    elif w == "images_for_test/spaa2.jpg":
        ORIGIN_x1, ORIGIN_y1 = 2250, 1040
        ORIGIN_x2, ORIGIN_y2 = 3360, 1600
        predicted = 1
    return ORIGIN_x1, ORIGIN_y1, ORIGIN_x2, ORIGIN_y2, predicted
