from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Project Configuration
# --------------------------------------------------

DATA_DIR = Path("Animal")
MODEL_PATH = Path("big_five_mobilenetv2.keras")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

CLASS_NAMES = [
    "Buffalo",
    "Elephant",
    "Leopard",
    "Lion",
    "Rhino"
]

# --------------------------------------------------
# Collect Image Paths and Labels/class-count
# --------------------------------------------------

VALID_EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".webp"}

image_paths = []
labels = []

for class_index, class_name in enumerate(CLASS_NAMES):
    class_folder = DATA_DIR / class_name

    if not class_folder.exists():
        raise FileNotFoundError(
            f"Missing class directory: {class_folder}"
        )

    for image_path in class_folder.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
            image_paths.append(str(image_path))
            labels.append(class_index)

image_paths = np.array(image_paths)
labels = np.array(labels)

print(f"Total images found: {len(image_paths)}")

for class_index, class_name in enumerate(CLASS_NAMES):
    class_count = np.sum(labels == class_index)
    print(f"{class_name}: {class_count} images")


# --------------------------------------------------
# Recreate Training, Validation and Test Split
# --------------------------------------------------

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.30,
    random_state=SEED,
    stratify=labels
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_labels
)

print("\nDataset Split:")
print(f"Training images: {len(train_paths)}")
print(f"Validation images: {len(val_paths)}")
print(f"Test images: {len(test_paths)}")

print("\nTest Set Distribution:")

for class_index, class_name in enumerate(CLASS_NAMES):
    class_count = np.sum(test_labels == class_index)
    print(f"{class_name}: {class_count} images")


# --------------------------------------------------
# Build Test Dataset
# --------------------------------------------------

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)

    return image, label


def make_test_dataset(paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


test_ds = make_test_dataset(test_paths, test_labels)

print("\nTest dataset created successfully.")


# --------------------------------------------------
# Check Trained Model
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

print(f"\nLoading trained model: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# --------------------------------------------------
# Generate Predictions on Test Set
# --------------------------------------------------

y_true = []
y_pred = []
y_probabilities = []

for images, batch_labels in test_ds:
    probabilities = model.predict(images, verbose=0)

    predictions = np.argmax(probabilities, axis=1)

    y_true.extend(batch_labels.numpy())
    y_pred.extend(predictions)
    y_probabilities.extend(probabilities)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_probabilities = np.array(y_probabilities)

print("\nPredictions completed successfully.")
print(f"Number of test images evaluated: {len(y_true)}")


# --------------------------------------------------
# Calculate Evaluation Metrics
# --------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)

print("\nOVERALL TEST PERFORMANCE")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

# --------------------------------------------------
# Per-Class Classification Report
# --------------------------------------------------

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
)

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

disp.plot(xticks_rotation=45)

plt.title("Big Five Species Classification - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "evaluation_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# --------------------------------------------------
# Individual Prediction Confidence
# --------------------------------------------------

print("\nINDIVIDUAL TEST IMAGE PREDICTIONS")

for i in range(len(test_paths)):
    actual_index = y_true[i]
    predicted_index = y_pred[i]

    actual_class = CLASS_NAMES[actual_index]
    predicted_class = CLASS_NAMES[predicted_index]

    confidence = y_probabilities[i][predicted_index] * 100

    image_name = Path(test_paths[i]).name

    result = "CORRECT" if actual_index == predicted_index else "INCORRECT"

    print(
        f"\nImage: {image_name}"
        f"\nActual: {actual_class}"
        f"\nPredicted: {predicted_class}"
        f"\nConfidence: {confidence:.2f}%"
        f"\nResult: {result}"
    )

# --------------------------------------------------
# Error and Confidence Analysis
# --------------------------------------------------

incorrect_indices = np.where(y_true != y_pred)[0]

print("\nERROR ANALYSIS")
print(f"Total incorrect predictions: {len(incorrect_indices)}")

if len(incorrect_indices) > 0:
    print("\nMisclassified Images:")

    for i in incorrect_indices:
        actual_class = CLASS_NAMES[y_true[i]]
        predicted_class = CLASS_NAMES[y_pred[i]]
        confidence = y_probabilities[i][y_pred[i]] * 100
        image_name = Path(test_paths[i]).name

        print(
            f"\nImage: {image_name}"
            f"\nActual: {actual_class}"
            f"\nPredicted: {predicted_class}"
            f"\nConfidence: {confidence:.2f}%"
        )

else:
    print("No images were misclassified in the test set.")

    print("\nLowest-Confidence Correct Predictions:")

    confidence_scores = np.max(y_probabilities, axis=1)
    lowest_confidence_indices = np.argsort(confidence_scores)[:3]

    for i in lowest_confidence_indices:
        actual_class = CLASS_NAMES[y_true[i]]
        predicted_class = CLASS_NAMES[y_pred[i]]
        confidence = confidence_scores[i] * 100
        image_name = Path(test_paths[i]).name

        print(
            f"\nImage: {image_name}"
            f"\nActual: {actual_class}"
            f"\nPredicted: {predicted_class}"
            f"\nConfidence: {confidence:.2f}%"
        )


# --------------------------------------------------
# Visualise Lowest-Confidence Predictions
# --------------------------------------------------

confidence_scores = np.max(y_probabilities, axis=1)
lowest_confidence_indices = np.argsort(confidence_scores)[:3]

plt.figure(figsize=(12, 4))

for plot_position, i in enumerate(lowest_confidence_indices):
    image = tf.io.read_file(test_paths[i])
    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    actual_class = CLASS_NAMES[y_true[i]]
    predicted_class = CLASS_NAMES[y_pred[i]]
    confidence = confidence_scores[i] * 100

    plt.subplot(1, 3, plot_position + 1)
    plt.imshow(image.numpy())
    plt.axis("off")

    plt.title(
        f"Actual: {actual_class}\n"
        f"Predicted: {predicted_class}\n"
        f"Confidence: {confidence:.2f}%"
    )

plt.suptitle("Lowest-Confidence Correct Predictions")
plt.tight_layout()

plt.savefig(
    "lowest_confidence_predictions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()