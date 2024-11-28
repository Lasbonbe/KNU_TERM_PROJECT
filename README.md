# Development of a Target Recognition Model for Missile Targeting System
<a href='https://github.com/shivamkapasia0' target="_blank"><img alt='PYTHON' src='https://img.shields.io/badge/3.12-100000?style=for-the-badge&logo=PYTHON&logoColor=white&labelColor=3950FF&color=3950FF'/></a>
<a href='https://github.com/shivamkapasia0' target="_blank"><img alt='TENSORFLOW' src='https://img.shields.io/badge/TENSORFLOW-100000?style=for-the-badge&logo=TENSORFLOW&logoColor=white&labelColor=black&color=black'/></a>
<a href='https://github.com/shivamkapasia0' target="_blank"><img alt='PYTORCH' src='https://img.shields.io/badge/PYTORCH-100000?style=for-the-badge&logo=PYTORCH&logoColor=white&labelColor=black&color=black'/></a>
<br>
![Last Commit](https://img.shields.io/github/last-commit/Lasbonbe/KNU_TERM_PROJECT)
![Code Size](https://img.shields.io/github/languages/code-size/Lasbonbe/KNU_TERM_PROJECT)
#### STUDENT ID : B202400543
#### NAME : 마티스 앙리 



<details>
<summary>

# **SUMMARY**

</summary>

- [Project Results and Overview](#project-results-and-overview)
- [Source Code](#source-code)
- [Performance Metrics](#performance-metrics)
- [Installation and Usage](#installation-and-usage)
- [References and Documentation](#references-and-documentation)
- [Issues and Contributions](#issues-and-contributions)
- [Future Work](#future-work)

</details>


![Missile Schema](assets/missileschema.png)

## Project Results and Overview
The key objective of this project is to develop a target recognition model for a missile targeting system. The model is designed to identify and classify 3 classes. 
The classes are:
- <a href='https://github.com/shivamkapasia0' target="_blank"><img alt='s' src='https://img.shields.io/badge/TANK-100000?style=for-the-badge&logo=s&logoColor=white&labelColor=FF0000&color=FF0000'/></a>
- <a href='https://github.com/shivamkapasia0' target="_blank"><img alt='s' src='https://img.shields.io/badge/SPAA (Self Propeled Anti Aircraft Vehicule)-100000?style=for-the-badge&logo=s&logoColor=white&labelColor=DE0CFF&color=DE0CFF'/></a> 
- <a href='https://github.com/shivamkapasia0' target="_blank"><img alt='s' src='https://img.shields.io/badge/MILITARY_VEHICULE (Other Type of military vehicule)-100000?style=for-the-badge&logo=s&logoColor=white&labelColor=FFAC0F&color=FFAC0F'/></a>

The project leverages deep learning techniques using PyTorch and TensorFlow frameworks.

### Key Results:
- Achieved an accuracy of **90%** on the validation dataset.
- Successfully converted the trained models (.tf / .pb .h5) to ONNX format for compatibility with OpenCV.
- Demonstrated real-time target recognition with a high degree of accuracy.

## Source Code

### FOLDERS

- `assets/`: Contains images and other assets used in the README.
- `models/`: Contains the trained PyTorch and TensorFlow models.
- `ONNX_models/`: Contains the converted ONNX models.
- `datasetV3/` : Contains the dataset used for training the model.
- `images_for_test/` : Contains the images used for testing the model.

### MAIN SCRIPTS
- `requirements.txt`: Contains the required libraries for the project.
----
- `DATASET_conversion.py`: This script converts the dataset to the required format for training.
- `DATASET_Download.py`: This script downloads the dataset from Kaggle.
----
- `PYTORCH_modeltraining.py`: This script trains the target recognition model using PyTorch.
- `PYTORCH_modelconversion.py`: This script converts the trained PyTorch model to ONNX format.
- `PYTORCH_modeltraining_YOLO.py`: This script trains the target recognition model using PyTorch with YOLO Dataset Format. ⚠️ DOES NOT WORK ⚠️
----
- `TENSORFLOW_modelconversion.py`: This script converts the trained PyTorch model to TensorFlow format.
- `TENSORFLOW_modeltraining.py`: This script trains the target recognition model using TensorFlow.
- `TENSORFLOW_freeze.py`: This script freezes the TensorFlow model for conversion to ONNX format. ⚠️BASICALY USELESS⚠️
----
- `OPENCV_Missile.py`: **MAIN INTEREST FILE** This script runs the target recognition system using OpenCV.

### TESTING/TROUBLESHOOTING SCRIPTS
- `TESTING.py`: The first iteration of OPENCV_Missile.py | Used mainly for testing new features and called functions to make the main file more visible.
- `TESTING_VERSION.py`: This script is used to test the different versions of the pip libraries.


## SETUP
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/your-repo.git
    cd your-repo
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```
   

## Performance Metrics
<a href='https://github.com/shivamkapasia0' target="_blank"><img alt='TENSORFLOW' src='https://img.shields.io/badge/TENSORFLOW-100000?style=for-the-badge&logo=TENSORFLOW&logoColor=white&labelColor=black&color=black'/></a>
<br>
![TENSORFLOW RESULTS](assets/tensor.gif)
<br>
<a href='https://github.com/shivamkapasia0' target="_blank"><img alt='PYTORCH' src='https://img.shields.io/badge/PYTORCH-100000?style=for-the-badge&logo=PYTORCH&logoColor=white&labelColor=black&color=black'/></a>

| Metric        | Value (TensorFlow) | Value (PyTorch) |
|---------------|--------------------|-----------------|
| Test Accuracy | 91.66%             | 85%             |
| Val Accuracy  | 56.28%             | 85%             |
| Precision     | 82%                | 83%             |
| Val Loss      | 4.8459             | N/A             |
| Test Loss     | 0.14               | N/A             |


### USAGE
1. Train the model using PyTorch:
    ```sh
    python PYTORCH_modeltraining.py
    ```

2. Convert the trained model to ONNX format:
    ```sh
    python PYTORCH_modelconversion.py
    ```

3. Run the target recognition system:
    ```sh
    python OPENCV_Missile.py
    ```

## References and Documentation
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [YOLO Paper](https://arxiv.org/abs/1506.02640)

## Issues and Contributions
### Known Issues
- ⚠️ Some scripts don't work as expected due to compatibility issues with the libraries.
- Limited dataset size may affect the generalization of the model.
- ⚠️ The bounding box display in the OpenCV isn't accurate when zooming


## Future Work
- Improve the model's performance in low-light conditions.
- Add Black And White Image Recognition. (Thermal Camera)
- Expand the dataset to include more diverse military equipment.
- Implement real-time target tracking capabilities.

