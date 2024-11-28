# Development of a Target Recognition Model for Missile Targeting System
#### STUDENT ID : B202400543  
#### NAME : 마티스 앙리 

![Missile Schema](assets/missileschema.png)

## Project Results and Overview
The key objective of this project is to develop a target recognition model for a missile targeting system. The model is designed to identify and classify 3 classes. 
The classes are:
- <a href='https://github.com/shivamkapasia0' target="_blank"><img alt='s' src='https://img.shields.io/badge/TANK-100000?style=for-the-badge&logo=s&logoColor=white&labelColor=FF0000&color=FF0000'/></a>
- <a href='https://github.com/shivamkapasia0' target="_blank"><img alt='s' src='https://img.shields.io/badge/SPAA (Self Propeled Anti Aircraft Vehicule)-100000?style=for-the-badge&logo=s&logoColor=white&labelColor=DE0CFF&color=DE0CFF'/></a> 
- <a href='https://github.com/shivamkapasia0' target="_blank"><img alt='s' src='https://img.shields.io/badge/MILITARY_VEHICULE (Other Type of military vehicule)-100000?style=for-the-badge&logo=s&logoColor=white&labelColor=FFAC0F&color=FFAC0F'/></a>

The project leverages deep learning techniques using PyTorch and TensorFlow frameworks.

### Key Results:
- Achieved an accuracy of 85% on the validation dataset.
- Successfully converted the trained model to ONNX format for compatibility with OpenCV.
- Demonstrated real-time target recognition with a high degree of accuracy.

## Source Code
The source code for this project is organized as follows:

### Instructions for Setup
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
The performance of the model was evaluated based on the following metrics:

| Metric        | Value (TensorFlow) | Value (PyTorch) |
|---------------|--------------------|-----------------|
| Accuracy      | 85%                | 85%             |
| Precision     | 82%                | 83%             |
| TEST accuracy | 85%                | 85%             |


## Installation and Usage
### Installation
1. Ensure you have Python 3.8+ installed.
2. Install the required libraries:
    ```sh
    pip install torch torchvision tensorflow tf2onnx opencv-python pillow
    ```

### Usage
1. Train the model using PyTorch:
    ```sh
    python PYTORCH_modeltraining_YOLO.py
    ```

2. Convert the trained model to ONNX format:
    ```sh
    python TENSORFLOW_modelconversion.py
    ```

3. Run the target recognition system:
    ```sh
    python run_target_recognition.py
    ```

## References and Documentation
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [YOLO Paper](https://arxiv.org/abs/1506.02640)

## Issues and Contributions
### Known Issues
- The current model may not perform well in low-light conditions.
- Limited dataset size may affect the generalization of the model.

### Contributions
We welcome contributions from the community. To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## Future Work
- Improve the model's performance in low-light conditions.
- Expand the dataset to include more diverse military equipment.
- Implement real-time target tracking capabilities.