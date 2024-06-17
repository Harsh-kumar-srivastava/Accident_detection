# Accident Detection Using Deep Learning

## Overview

This repository contains code for an accident detection model using a Convolutional Neural Network (CNN). The model is designed to classify video frames into accident or non-accident categories. The implementation includes data preprocessing, model creation, training, and evaluation.

## Table of Contents

### 1. Requirements
### 2. Dataset Preparation
### 3. Model Architecture
### 4. Training and Evaluation
### 5. Usage
### 6. Results
### 7. Credits

## Requirements
To run the code, you need the following libraries and dependencies:

### Python 
### OpenCV
### NumPy
### TensorFlow

You can install the required libraries using pip:
```bash
pip install opencv-python numpy tensorflow
```

## Dataset Preparation

1. Accident Dataset: Place the accident video files in the directory: /content/drive/MyDrive/datasets/Accident.
2. Non-Accident Dataset: Place the non-accident video files in the directory /content/drive/MyDrive/datasets/No Accident.

## Model Architecture

The model uses 3D convolutional layers to capture the temporal features of video frames. The architecture is as follows:

1. Conv3D layer with 4 filters, kernel size of (3, 3, 3), ReLU activation, followed by MaxPooling3D and Dropout layers.
2. Conv3D layer with 8 filters, kernel size of (3, 3, 3), ReLU activation, followed by MaxPooling3D and Dropout layers.
3. Conv3D layer with 16 filters, kernel size of (3, 3, 3), ReLU activation, followed by MaxPooling3D and Dropout layers.
4. Conv3D layer with 32 filters, kernel size of (3, 3, 3), ReLU activation, followed by MaxPooling3D and Dropout layers.
5. Flatten layer to convert the 3D output to 1D.
6. Dense layer with a single unit and sigmoid activation for binary classification.

## Training and Evaluation

The dataset is split into training and testing sets with a 70-30 ratio. The model is compiled using the Adam optimizer and binary cross-entropy loss function. Accuracy is used as the evaluation metric.

## Usage

1. Clone the repository: 
```bash
git clone https://github.com/Harsh-kumar-srivastava/Accident_detection.git
```
2. Ensure your dataset directories are correctly set up as described in the Dataset Preparation section.
3. Run the script to train and evaluate the model:
```bash
python accident_detection.py
```
The script will output the shapes of the training and testing datasets, and the model summary. After training, the model will achieve around 80% accuracy.

## Results

The final model achieved an accuracy of approximately 80% on the test dataset.

## Credits

This project was developed by:
- Harsh Srivastava
- Pawan Kumar

For further details and updates, please refer to the code comments and documentation within the script files. If you encounter any issues or have questions, feel free to open an issue in the repository.
