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

| Class | Animal   |
| ----- | -------- |
| 0     | Buffalo  |
| 1     | Elephant |
| 2     | Leopard  |
| 3     | Lion     |
| 4     | Rhino    |

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
```

---

## 8. Project Structure

```text
BigFiveApplication/                  (repository root)
├── README.md
├── requirements.txt
└── BigFiveApplication/
    ├── Animal/                      dataset: one folder per class, 15 images each
    │   ├── Buffalo/  Elephant/  Leopard/  Lion/  Rhino/
    ├── training.py                  Member 2: trains MobileNetV2, saves big_five_mobilenetv2.keras
    ├── evaluation.py                Member 3: metrics, confusion matrix, error analysis
    ├── predict.py                   Member 4: prediction module and command line tool
    ├── app.py                       Member 4: Gradio web interface
    ├── examples/                    one unseen test image per class, used in the demo
    └── big_five_mobilenetv2.keras   trained model (created by training.py)
```

All scripts are run from inside the inner `BigFiveApplication/` folder, because they use the relative paths `Animal/` and `big_five_mobilenetv2.keras`.

---

## 9. Requirements

- Python 3.10 or newer
- The packages in `requirements.txt` (TensorFlow, NumPy, Pillow, scikit-learn, Matplotlib, Gradio)
- About 2 GB of free disk space for TensorFlow
- Internet access the first time `training.py` runs, to download the ImageNet weights for MobileNetV2
- A GPU is not required. Training on this dataset takes about 2 to 3 minutes on a normal CPU, and less on Google Colab.

---

## 10. Installation

### Option A: Google Colab

```python
!git clone https://github.com/LuckyMkhetyeva/BigFiveApplication.git
%cd BigFiveApplication/BigFiveApplication
!pip install -q gradio
```

TensorFlow, scikit-learn and Matplotlib are already installed on Colab.

### Option B: Local machine

```bash
git clone https://github.com/LuckyMkhetyeva/BigFiveApplication.git
cd BigFiveApplication
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # Mac/Linux
pip install -r requirements.txt
cd BigFiveApplication
```

---

## 11. Dataset

The dataset is included in the repository in `BigFiveApplication/Animal/`. It contains 75 images, 15 for each of the five classes. The folder names are the class labels and must stay exactly as they are: `Buffalo`, `Elephant`, `Leopard`, `Lion`, `Rhino`.

`training.py` and `evaluation.py` split the images 70/15/15 into training (52), validation (11) and test (12) sets. The split is stratified and uses a fixed random seed (42), so both scripts always produce the same split.

---

## 12. How to Train

```bash
python training.py
```

The script trains the classifier in two stages (frozen MobileNetV2, then fine tuning of its last 30 layers), prints the test results and saves:

- `big_five_mobilenetv2.keras`, the final model used by the application
- `accuracy_curve.png`, `loss_curve.png` and `confusion_matrix.png`

## 13. How to Evaluate

```bash
python evaluation.py
```

This loads `big_five_mobilenetv2.keras`, recreates the same test split and saves `evaluation_results.txt`, `evaluation_confusion_matrix.png` and `lowest_confidence_predictions.png`.

---

## 14. How to Run a Prediction

`big_five_mobilenetv2.keras` must be in the same folder. If it is not there, run `training.py` first. Without it, the application stops with an error message explaining that the model file is missing.

### Command line

```bash
python predict.py examples/lion.jfif --all   # also shows the score for every class
python app.py --share                        # also creates a public link (use in Colab)
```

Example output:

```text
Animal: Lion
Confidence: 97.9%
```

### Web interface (Gradio)

```bash
python app.py
python app.py --share
```

Upload an image or click one of the examples. The application shows the predicted species, the confidence and a bar for every class.

### From Python or a notebook

```python
from predict import AnimalClassifier, format_result

clf = AnimalClassifier("big_five_mobilenetv2.keras")
result = clf.predict("examples/leopard.jfif")
print(format_result(result))
print(result["scores"])
```

---

## 15. Team Responsibilities

| Member   | Responsibility                                                                      |
| -------- | ----------------------------------------------------------------------------------- |
| Member 1 | Dataset and data preparation                                                        |
| Member 2 | MobileNetV2 model and training (`training.py`)                                      |
| Member 3 | Evaluation and testing (`evaluation.py`)                                            |
| Member 4 | Prediction application, integration and setup instructions (`predict.py`, `app.py`) |
