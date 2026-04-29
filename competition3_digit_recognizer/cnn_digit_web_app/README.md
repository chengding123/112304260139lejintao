# CNN 手写数字识别 Web App

这是报告中“实验二：模型封装为 Web 部署”和“实验三：交互式手写识别系统”对应的 Gradio 网站。
模型来自 Kaggle Digit Recognizer 训练结果，当前页面只保留在线手写识别输入。

## 功能

- 使用网页手写板直接书写 0-9 数字。
- 手写板只提供画笔、擦除、清空和识别功能，避免复杂图片编辑工具干扰。
- 自动灰度化、缩放到 `28x28`、适配黑白背景。
- 显示预测数字、置信度和 Top-3 概率。

## 本地运行

```powershell
py -m pip install -r requirements.txt
py app.py
```

启动后访问：`http://127.0.0.1:7860`

## Render 部署

仓库根目录已提供 `render.yaml`。在 Render 中选择本仓库 Blueprint 或 Web Service 后，配置如下：

- Root Directory: `competition3_digit_recognizer/cnn_digit_web_app`
- Build Command: `pip install -r requirements.txt`
- Start Command: `python app.py`

部署成功后访问 Render 分配的 `https://*.onrender.com` 地址。
