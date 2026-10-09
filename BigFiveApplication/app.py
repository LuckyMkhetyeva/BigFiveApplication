"""
app.py - Big Five animal classifier: Gradio web interface 

Run locally:     python app.py
Run in Colab:    !python app.py --share     

Upload an image -> preprocess -> MobileNetV2 -> species + confidence.
"""

import argparse
from pathlib import Path

import gradio as gr

from predict import (AnimalClassifier, PredictionError, DEFAULT_MODEL_PATH,
                     LOW_CONFIDENCE_THRESHOLD, ALLOWED_EXTENSIONS)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
parser.add_argument("--port", type=int, default=7860)
args, _ = parser.parse_known_args()

try:
    classifier = AnimalClassifier(args.model)
except PredictionError as exc:
    raise SystemExit(f"Error: {exc}")


def classify(image):
    if image is None:
        return {}, "Please upload an image first."
    try:
        result = classifier.predict(image)
    except PredictionError as exc:
        return {}, f"**Error:** {exc}"

    text = (f"## {result['label']}\n"
            f"**Confidence:** {result['confidence'] * 100:.1f}%")
    if result["low_confidence"]:
        text += (f"\n\n⚠️ Confidence is below {LOW_CONFIDENCE_THRESHOLD:.0%}. "
                 "The image may be unclear, show several animals, or show an "
                 "animal that is not one of the Big Five.")
    return result["scores"], text


examples_dir = Path("examples")
examples = sorted(str(p) for p in examples_dir.glob("*")
                  if p.suffix.lower() in ALLOWED_EXTENSIONS) if examples_dir.exists() else None

with gr.Blocks(title="Big Five Animal Classifier") as demo:
    gr.Markdown(
        "# Big Five Animal Classifier\n"
        "Upload a photo of a **lion, elephant, buffalo, rhinoceros or leopard**. "
        "The MobileNetV2 model predicts the species and shows how confident it is."
    )
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Animal image")
            with gr.Row():
                clear_btn = gr.ClearButton(value="Clear")
                submit_btn = gr.Button("Classify", variant="primary")
        with gr.Column():
            summary_out = gr.Markdown()
            label_out = gr.Label(num_top_classes=5, label="Scores for each class")

    if examples:
        gr.Examples(examples=examples, inputs=image_in)

    submit_btn.click(classify, inputs=image_in, outputs=[label_out, summary_out])
    image_in.upload(classify, inputs=image_in, outputs=[label_out, summary_out])
    clear_btn.add([image_in, label_out, summary_out])

if __name__ == "__main__":
    demo.launch(share=args.share, server_port=args.port)
