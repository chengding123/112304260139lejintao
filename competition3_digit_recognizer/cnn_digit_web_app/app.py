import base64
import os
from io import BytesIO
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps


IMAGE_SIZE = 28
NUM_CLASSES = 10
MODEL_PATH = Path("model.pth")
EXPERIMENT_SCORE = "0.99571"
LOCAL_TEST_ACC = "99.17%"


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
        return "请先上传或绘制一个数字", {}

    tensor = preprocess(image)
    logits = MODEL(tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0).numpy()
    prediction = int(probabilities.argmax())
    label_probs = {str(i): float(probabilities[i]) for i in range(NUM_CLASSES)}
    confidence = float(probabilities[prediction])
    result = f"预测结果：{prediction}    置信度：{confidence:.2%}"
    return result, label_probs


@torch.no_grad()
def predict_sketch(image):
    if image is None:
        return "请先在手写板中写一个数字", {}

    if isinstance(image, dict):
        image = image.get("composite") or image.get("image")

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    return predict_uploaded(image)


@torch.no_grad()
def predict_canvas(data_url: str):
    if not data_url:
        return "请先在手写板中写一个数字", {}

    try:
        _, encoded = data_url.split(",", 1)
        image = Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
    except Exception:
        return "手写板数据读取失败，请清空后重新书写", {}

    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if float((gray < 245).mean()) < 0.005:
        return "请先在手写板中写一个数字", {}

    return predict_uploaded(image)


CSS = """
:root {
    --ink: #18212f;
    --muted: #697386;
    --paper: #fff7e8;
    --cream: #fffdf7;
    --accent: #e95f37;
    --accent-2: #195f68;
    --gold: #f5b84b;
}
body, .gradio-container {
    background:
        radial-gradient(circle at 12% 10%, rgba(245, 184, 75, 0.30), transparent 25rem),
        radial-gradient(circle at 90% 6%, rgba(25, 95, 104, 0.22), transparent 28rem),
        linear-gradient(145deg, #fff7e8 0%, #f2fbf7 52%, #edf4ff 100%) !important;
    color: var(--ink);
    font-family: Georgia, "Times New Roman", "Noto Serif SC", serif;
}
.gradio-container {
    max-width: 1180px !important;
}
.main {
    animation: rise-in .7s ease both;
}
@keyframes rise-in {
    from { opacity: 0; transform: translateY(18px); }
    to { opacity: 1; transform: translateY(0); }
}
.hero {
    position: relative;
    overflow: hidden;
    padding: 34px;
    border-radius: 34px;
    color: #ffffff;
    background:
        radial-gradient(circle at 82% 18%, rgba(245, 184, 75, .48), transparent 18rem),
        linear-gradient(135deg, rgba(19, 29, 43, 0.98), rgba(25, 95, 104, 0.95)),
        repeating-linear-gradient(45deg, rgba(255,255,255,.08) 0 1px, transparent 1px 12px);
    box-shadow: 0 30px 80px rgba(24, 33, 47, 0.24);
}
.hero:after {
    content: "0 1 2 3 4 5 6 7 8 9";
    position: absolute;
    right: -24px;
    bottom: -18px;
    color: rgba(255,255,255,.10);
    font-size: clamp(3rem, 9vw, 7rem);
    font-weight: 800;
    letter-spacing: -.08em;
    white-space: nowrap;
}
.hero h1 {
    margin: 0 0 10px;
    font-size: clamp(2.1rem, 5vw, 4.4rem);
    line-height: 1.03;
    letter-spacing: -0.05em;
    font-family: "Trebuchet MS", "Noto Sans SC", sans-serif;
}
.hero p {
    color: rgba(255,255,255,.86);
    max-width: 850px;
    font-size: 1.05rem;
}
.badge {
    display: inline-flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 18px;
    padding: 8px 12px;
    border: 1px solid rgba(255,255,255,.24);
    border-radius: 999px;
    background: rgba(255,255,255,.12);
    color: rgba(255,255,255,.92);
    font-size: .92rem;
}
.metric-card {
    min-height: 112px;
    padding: 20px;
    border-radius: 24px;
    background: rgba(255,253,247,.86);
    border: 1px solid rgba(24, 33, 47, .10);
    box-shadow: 0 18px 42px rgba(24, 33, 47, .08);
    transition: transform .18s ease, box-shadow .18s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 24px 54px rgba(24, 33, 47, .13);
}
.metric-card strong {
    display: block;
    margin: 4px 0;
    font-size: 1.9rem;
    color: var(--accent-2);
    font-family: "Trebuchet MS", "Noto Sans SC", sans-serif;
}
.metric-card span {
    color: var(--muted);
    font-size: .92rem;
}
.metric-card small {
    color: var(--muted);
}
.panel-note {
    padding: 16px 18px;
    border-radius: 20px;
    background: rgba(255, 253, 247, .72);
    border-left: 5px solid var(--accent);
}
.tip {
    color: var(--muted);
    font-size: .95rem;
}
.gr-button-primary {
    background: linear-gradient(135deg, var(--accent), var(--gold)) !important;
    border: none !important;
    color: #1a2433 !important;
    font-weight: 700 !important;
}
.tabs button {
    font-weight: 700 !important;
}
.draw-card {
    padding: 20px;
    border-radius: 28px;
    background: rgba(255,253,247,.86);
    border: 1px solid rgba(24, 33, 47, .10);
    box-shadow: 0 20px 50px rgba(24, 33, 47, .10);
}
#digit-canvas {
    display: block;
    width: min(100%, 430px);
    aspect-ratio: 1 / 1;
    margin: 0 auto 16px;
    border-radius: 24px;
    background: #ffffff;
    border: 2px solid rgba(24, 33, 47, .15);
    box-shadow: inset 0 0 0 1px rgba(255,255,255,.9);
    cursor: crosshair;
    touch-action: none;
}
.canvas-toolbar {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: center;
}
.tool-btn {
    min-width: 92px;
    padding: 10px 16px;
    border: 1px solid rgba(24, 33, 47, .14);
    border-radius: 999px;
    background: #ffffff;
    color: var(--ink);
    cursor: pointer;
    font-weight: 700;
}
.tool-btn.active {
    background: linear-gradient(135deg, var(--accent), var(--gold));
    border-color: transparent;
    color: #17202a;
}
"""


APP_JS = """
function () {
    const setupCanvas = () => {
        const canvas = document.getElementById("digit-canvas");
        if (!canvas || canvas.dataset.ready === "true") {
            return Boolean(canvas && canvas.dataset.ready === "true");
        }
        canvas.dataset.ready = "true";
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        const brushButton = document.getElementById("brush-tool");
        const eraserButton = document.getElementById("eraser-tool");
        const clearButton = document.getElementById("clear-canvas");
        if (!brushButton || !eraserButton || !clearButton) {
            canvas.dataset.ready = "false";
            return false;
        }
        let drawing = false;
        let tool = "brush";

        const fillWhite = () => {
            ctx.save();
            ctx.globalCompositeOperation = "source-over";
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.restore();
        };

        const setTool = (nextTool) => {
            tool = nextTool;
            brushButton.classList.toggle("active", tool === "brush");
            eraserButton.classList.toggle("active", tool === "eraser");
        };

        const point = (event) => {
            const rect = canvas.getBoundingClientRect();
            return {
                x: (event.clientX - rect.left) * canvas.width / rect.width,
                y: (event.clientY - rect.top) * canvas.height / rect.height
            };
        };

        const strokeTo = (event) => {
            if (!drawing) return;
            event.preventDefault();
            const pos = point(event);
            ctx.lineTo(pos.x, pos.y);
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
            ctx.lineWidth = tool === "brush" ? 28 : 38;
            ctx.strokeStyle = tool === "brush" ? "#000000" : "#ffffff";
            ctx.stroke();
        };

        canvas.addEventListener("pointerdown", (event) => {
            event.preventDefault();
            drawing = true;
            canvas.setPointerCapture(event.pointerId);
            const pos = point(event);
            ctx.beginPath();
            ctx.moveTo(pos.x, pos.y);
            strokeTo(event);
        });
        canvas.addEventListener("pointermove", strokeTo);
        canvas.addEventListener("pointerup", () => drawing = false);
        canvas.addEventListener("pointercancel", () => drawing = false);
        brushButton.addEventListener("click", () => setTool("brush"));
        eraserButton.addEventListener("click", () => setTool("eraser"));
        clearButton.addEventListener("click", fillWhite);

        fillWhite();
        setTool("brush");
        return true;
    };

    let attempts = 0;
    const timer = setInterval(() => {
        attempts += 1;
        if (setupCanvas() || attempts > 20) {
            clearInterval(timer);
        }
    }, 200);
}
"""


with gr.Blocks(title="CNN 手写数字识别系统", css=CSS, js=APP_JS) as demo:
    gr.HTML(
        f"""
        <section class="hero">
            <div class="badge">Kaggle Digit Recognizer · PyTorch CNN · Gradio Web</div>
            <h1>CNN 手写数字识别系统</h1>
            <p>
                基于 Kaggle Digit Recognizer 数据集训练的 PyTorch CNN 模型。页面只保留在线手写输入功能，
                自动完成 28x28 归一化、黑白背景判断，并输出识别结果和 Top-3 预测概率。
            </p>
        </section>
        """
    )

    with gr.Row():
        gr.HTML(f'<div class="metric-card"><span>Kaggle Score</span><strong>{EXPERIMENT_SCORE}</strong><small>非外部数据提交</small></div>')
        gr.HTML(f'<div class="metric-card"><span>本地最佳实验</span><strong>{LOCAL_TEST_ACC}</strong><small>Exp4: Adam + 增强 + Early Stopping</small></div>')
        gr.HTML('<div class="metric-card"><span>模型结构</span><strong>CNN</strong><small>Conv-BN-ReLU + Dropout</small></div>')

    gr.HTML(
        """
        <div class="panel-note">
            <strong>使用说明</strong>：在手写板中写一个 0-9 的单个数字，数字尽量居中、线条尽量清晰。
            系统会自动适配白底黑字和黑底白字图片。
        </div>
        """
    )

    gr.HTML('<p class="tip">手写板只保留画笔和擦除功能。线条粗一点、数字居中时效果更稳定。</p>')
    with gr.Row():
        gr.HTML(
            """
            <div class="draw-card">
                <canvas id="digit-canvas" width="420" height="420" aria-label="手写数字画板"></canvas>
                <div class="canvas-toolbar">
                    <button id="brush-tool" class="tool-btn active" type="button">画笔</button>
                    <button id="eraser-tool" class="tool-btn" type="button">擦除</button>
                    <button id="clear-canvas" class="tool-btn" type="button">清空</button>
                </div>
            </div>
            """
        )
        with gr.Column():
            sketch_label = gr.Textbox(label="识别结果", interactive=False)
            sketch_probs = gr.Label(label="Top-3 概率", num_top_classes=3)
            sketch_button = gr.Button("识别手写数字", variant="primary")
            canvas_data = gr.Textbox(visible=False)
    sketch_button.click(
        predict_canvas,
        inputs=canvas_data,
        outputs=[sketch_label, sketch_probs],
        js='() => document.getElementById("digit-canvas").toDataURL("image/png")',
    )

    gr.HTML(
        """
        <div class="panel-note">
            <strong>实验对应关系</strong>：该网站重点展示报告中的“实验三：交互式手写识别系统”，
            使用实验一训练好的 CNN 模型完成在线手写数字识别。
        </div>
        """
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
