"""
predict.py - Big Five animal classifier: prediction module 

Loads the trained MobileNetV2 model (big_five_mobilenetv2.keras, produced by
training.py), preprocesses a
user image and returns the predicted species with a confidence score.

Usage from the command line:
    python predict.py path/to/image.jpg
    python predict.py path/to/image.jpg --model big_five_mobilenetv2.keras

Usage from Python / Colab:
    from predict import AnimalClassifier
    clf = AnimalClassifier("big_five_mobilenetv2.keras")
    result = clf.predict("lion.jpg")
    print(result["label"], result["confidence"])

"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError


DEFAULT_MODEL_PATH = "big_five_mobilenetv2.keras"
CLASS_NAMES_PATH = "class_names.json" 


DEFAULT_CLASS_NAMES = ["Buffalo", "Elephant", "Leopard", "Lion", "Rhino"]

DISPLAY_NAMES = {
    "buffalo": "Buffalo",
    "elephant": "Elephant",
    "leopard": "Leopard",
    "lion": "Lion",
    "rhino": "Rhinoceros",
    "rhinoceros": "Rhinoceros",
}

IMG_SIZE = (224, 224)

PREPROCESS_IN_MODEL = os.environ.get("PREPROCESS_IN_MODEL", "1") == "1"


LOW_CONFIDENCE_THRESHOLD = 0.60

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".webp"}


class PredictionError(Exception):
    """Raised when an image cannot be classified."""


def load_class_names(path: str = CLASS_NAMES_PATH) -> list[str]:
    """Use class_names.json from training if it exists, else the default order."""
    p = Path(path)
    if p.exists():
        with p.open() as f:
            names = json.load(f)
        if isinstance(names, list) and all(isinstance(n, str) for n in names):
            return names
    return list(DEFAULT_CLASS_NAMES)


def display_name(raw: str) -> str:
    return DISPLAY_NAMES.get(raw.lower(), raw.replace("_", " ").title())


def load_image(source) -> Image.Image:
    """Accept a file path, a PIL image or a numpy array; return an RGB PIL image."""
    if source is None:
        raise PredictionError("No image was provided. Please upload an image.")

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise PredictionError(f"File not found: {path}")
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise PredictionError(
                f"Unsupported file type '{path.suffix}'. "
                f"Use one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )
        try:
            img = Image.open(path)
            img.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise PredictionError("The file could not be opened as an image. "
                                  "It may be corrupted.") from exc
    elif isinstance(source, Image.Image):
        img = source
    elif isinstance(source, np.ndarray):
        img = Image.fromarray(source.astype("uint8"))
    else:
        raise PredictionError(f"Unsupported input type: {type(source).__name__}")

   
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def preprocess(img: Image.Image) -> np.ndarray:
    """Resize to 224x224 and build a batch of one, ready for the model.

    Uses tf.image.resize (bilinear), the same call training.py uses, so the
    model sees images prepared exactly as in training.
    """
    arr = np.asarray(img, dtype="float32")
    arr = tf.image.resize(arr, IMG_SIZE).numpy()
    if not PREPROCESS_IN_MODEL:
    
        arr = arr / 127.5 - 1.0
    return np.expand_dims(arr, axis=0)  


class AnimalClassifier:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH,
                 class_names: list[str] | None = None):
        self.model_path = model_path
        self.class_names = class_names or load_class_names()

        if not Path(model_path).exists():
            raise PredictionError(
                f"Model file not found: {model_path}. "
                "Run training.py first, or copy the trained model into this folder."
            )

        self.model = tf.keras.models.load_model(model_path)
        n_out = self.model.output_shape[-1]
        if n_out != len(self.class_names):
            raise PredictionError(
                f"Model has {n_out} outputs but {len(self.class_names)} "
                f"class names were given: {self.class_names}"
            )

    def predict(self, source) -> dict:
        """Classify one image. Returns label, confidence and all class scores."""
        img = load_image(source)
        batch = preprocess(img)

        probs = np.asarray(self.model.predict(batch, verbose=0)[0], dtype="float64")
       
        if probs.min() < 0 or not np.isclose(probs.sum(), 1.0, atol=1e-3):
            e = np.exp(probs - probs.max())
            probs = e / e.sum()

        top = int(np.argmax(probs))
        confidence = float(probs[top])
        scores = {display_name(c): float(p) for c, p in zip(self.class_names, probs)}
        scores = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))

        return {
            "label": display_name(self.class_names[top]),
            "confidence": confidence,
            "scores": scores,
            "low_confidence": confidence < LOW_CONFIDENCE_THRESHOLD,
        }


def format_result(result: dict) -> str:
    lines = [f"Animal: {result['label']}",
             f"Confidence: {result['confidence'] * 100:.1f}%"]
    if result["low_confidence"]:
        lines.append("Warning: low confidence. The image may be unclear or may "
                     "not show one of the Big Five.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Classify a Big Five animal image.")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to the .keras model")
    parser.add_argument("--all", action="store_true", help="Show scores for all classes")
    args = parser.parse_args()

    try:
        clf = AnimalClassifier(args.model)
        result = clf.predict(args.image)
    except PredictionError as exc:
        raise SystemExit(f"Error: {exc}")

    print(format_result(result))
    if args.all:
        for name, p in result["scores"].items():
            print(f"  {name:<11} {p * 100:5.1f}%")


if __name__ == "__main__":
    main()
