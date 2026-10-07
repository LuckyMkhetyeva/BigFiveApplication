# AIE580S-DSE580S Topic 5
# African Big Five Image Classification using CNN + MobileNetV2 Transfer Learning

# Classes:
# 0 Buffalo
# 1 Elephant
# 2 Leopard
# 3 Lion
# 4 Rhino

# Place the downloaded five-class dataset under:
# data/
#     Buffalo/
#     Elephant/
#     Leopard/
#     Lion/
#     Rhino/

# The script performs:
# 1. Dataset inspection and class counts
# 2. Train/validation/test split
# 3. Training-only augmentation
# 4. MobileNetV2 transfer learning
# 5. Optional fine-tuning
# 6. Evaluation with accuracy, precision, recall, F1 and confusion matrix
# 7. Saving the final model
# """

import os
import random
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = Path("Animal")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5
CLASS_NAMES = ["Buffalo", "Elephant", "Leopard", "Lion", "Rhino"]

# ---------------------------------------------------------------------
# 1. Collect file paths and labels
# ---------------------------------------------------------------------
valid_ext = {".jpg", ".jfif", ".jpeg", ".png", ".bmp", ".webp"}
paths, labels = [], []

for label, class_name in enumerate(CLASS_NAMES):
    class_dir = DATA_DIR / class_name
    if not class_dir.exists():
        raise FileNotFoundError(f"Missing class directory: {class_dir}")
    for p in class_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in valid_ext:
            paths.append(str(p))
            labels.append(label)

paths = np.array(paths)
labels = np.array(labels)

print("\nCLASS DISTRIBUTION")
for i, name in enumerate(CLASS_NAMES):
    print(f"{name}: {(labels == i).sum()}")

# ---------------------------------------------------------------------
# 2. Stratified 70/15/15 split
# ---------------------------------------------------------------------
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    paths, labels, test_size=0.30, random_state=SEED, stratify=labels
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.50, random_state=SEED, stratify=temp_labels
)

print(f"\nTrain: {len(train_paths)}")
print(f"Validation: {len(val_paths)}")
print(f"Test: {len(test_paths)}")

# ---------------------------------------------------------------------
# 3. Build tf.data datasets
# ---------------------------------------------------------------------
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label

def make_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_paths, train_labels, shuffle=True)
val_ds = make_dataset(val_paths, val_labels)
test_ds = make_dataset(test_paths, test_labels)

# ---------------------------------------------------------------------
# 4. Training-only augmentation
# ---------------------------------------------------------------------
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
], name="data_augmentation")

# ---------------------------------------------------------------------
# 5. MobileNetV2 transfer-learning model
# ---------------------------------------------------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.30)(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.20)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_mobilenetv2.keras", monitor="val_accuracy",
        save_best_only=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks
)

# ---------------------------------------------------------------------
# 6. Optional fine-tuning
# ---------------------------------------------------------------------
base_model.trainable = True

# Freeze earlier layers and fine-tune only the final part.
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

fine_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)

# ---------------------------------------------------------------------
# 7. Test evaluation
# ---------------------------------------------------------------------
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
print(f"\nTest accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

y_true = []
y_pred = []

for images, batch_labels in test_ds:
    probs = model.predict(images, verbose=0)
    predictions = np.argmax(probs, axis=1)
    y_true.extend(batch_labels.numpy())
    y_pred.extend(predictions)

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_true, y_pred,
    target_names=CLASS_NAMES,
    digits=4
))

# ---------------------------------------------------------------------
# 8. Confusion matrix
# ---------------------------------------------------------------------
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(xticks_rotation=45)
plt.title("MobileNetV2 Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# ---------------------------------------------------------------------
# 9. Training curves
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_curve.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300)
plt.show()

# ---------------------------------------------------------------------
# 10. Save final model
# ---------------------------------------------------------------------
model.save("big_five_mobilenetv2.keras")
print("\nSaved model: big_five_mobilenetv2.keras")
