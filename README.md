# Big Five Animal Classifier Using CNN and MobileNetV2

## AIE580S – DSE580S Group Assignment

### Project Title

**African Big Five Animal Species Classification Using a Convolutional Neural Network (CNN) with Transfer Learning Using MobileNetV2**

---

## 1. Project Overview

This project develops an image classification system capable of identifying African Big Five animal species from images.

The system uses a Convolutional Neural Network (CNN) with the MobileNetV2 architecture and transfer learning. A pretrained MobileNetV2 model is used as the feature extraction backbone and adapted to classify images into five animal categories.

The five target classes are:

- Buffalo
- Elephant
- Leopard
- Lion
- Rhino

The system accepts an animal image as input and produces a predicted animal species together with a confidence score.

---

## 2. Problem Statement

Identifying animal species from photographs can be challenging because images may contain different backgrounds, lighting conditions, viewpoints, distances and levels of visibility.

This project addresses this problem by developing an automated image classification system that uses deep learning to recognise African Big Five animal species.

The project is based on Topic 5 of the AIE580S–DSE580S group assignment.

---

## 3. Aim

The aim of this project is to develop and evaluate a CNN-based image classification system using MobileNetV2 transfer learning to classify images of the African Big Five.

---

## 4. Objectives

The project aims to:

1. Obtain and prepare a labelled dataset containing the five target animal classes.
2. Perform data cleaning and preprocessing.
3. Explore the distribution and quality of the dataset.
4. Implement a CNN-based classifier using MobileNetV2.
5. Apply transfer learning using pretrained ImageNet weights.
6. Train and fine-tune the classification model.
7. Evaluate the model using appropriate performance metrics.
8. Produce a predicted animal class and confidence score for a new image.

---

## 5. Target Classes

| Class | Animal |
|---|---|
| 0 | Buffalo |
| 1 | Elephant |
| 2 | Leopard |
| 3 | Lion |
| 4 | Rhino |

---

## 6. Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Matplotlib
- Scikit-learn
- Pillow
- Jupyter Notebook / VS Code
- GitHub

---

## 7. Model Architecture

The proposed model follows this architecture:

```text
Input Image
     |
     v
224 × 224 × 3
     |
     v
MobileNetV2
(ImageNet pretrained weights)
     |
     v
Global Average Pooling
     |
     v
Dropout
     |
     v
Dense Layer - 128 neurons
     |
     v
Dropout
     |
     v
Dense Layer - 5 neurons
     |
     v
Softmax
     |
     v
Predicted Animal + Confidence
