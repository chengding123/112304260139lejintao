from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps


IMAGE_SIZE = 28
NUM_CLASSES = 10
MODEL_PATH = Path("model.pth")


class DigitCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.10),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.15),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.35),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = DigitCNN()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    mean = float(checkpoint.get("mean", 0.1310153379))
    std = float(checkpoint.get("std", 0.3085411559))
    return model, mean, std


MODEL, MEAN, STD = load_model()


def preprocess(image: Image.Image) -> torch.Tensor:
    image = image.convert("L")
    image = ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0

    # Uploaded handwriting is often dark strokes on light background; MNIST uses the opposite.
    if array.mean() > 0.5:
        array = 1.0 - array

    array = (array - MEAN) / STD
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).float()
    return tensor


@torch.no_grad()
def predict_uploaded(image: Image.Image):
    if image is None:
        return None, {}

    tensor = preprocess(image)
    logits = MODEL(tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0).numpy()
    prediction = int(probabilities.argmax())
    label_probs = {str(i): float(probabilities[i]) for i in range(NUM_CLASSES)}
    return prediction, label_probs


@torch.no_grad()
def predict_sketch(image):
    if image is None:
        return None, {}

    if isinstance(image, dict):
        image = image.get("composite") or image.get("image")

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    return predict_uploaded(image)


with gr.Blocks(title="CNN Handwritten Digit Recognizer") as demo:
    gr.Markdown("# CNN Handwritten Digit Recognizer")

    with gr.Tab("Upload"):
        upload = gr.Image(type="pil", label="Upload digit image")
        upload_label = gr.Number(label="Prediction", precision=0)
        upload_probs = gr.Label(label="Class probabilities", num_top_classes=3)
        upload.change(predict_uploaded, inputs=upload, outputs=[upload_label, upload_probs])

    with gr.Tab("Sketchpad"):
        sketch = gr.ImageEditor(type="pil", label="Write a digit")
        sketch_label = gr.Number(label="Prediction", precision=0)
        sketch_probs = gr.Label(label="Top-3 probabilities", num_top_classes=3)
        sketch.change(predict_sketch, inputs=sketch, outputs=[sketch_label, sketch_probs])


if __name__ == "__main__":
    demo.launch()

