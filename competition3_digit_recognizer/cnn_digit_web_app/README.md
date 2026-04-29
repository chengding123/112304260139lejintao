# CNN Handwritten Digit Recognizer Web App

This is a Gradio app for the Kaggle Digit Recognizer CNN model.

## Files

- `app.py`: Gradio application.
- `model.pth`: trained PyTorch checkpoint.
- `requirements.txt`: runtime dependencies.

## Run Locally

```powershell
py -m pip install -r requirements.txt
py app.py
```

## Features

- Upload a digit image and classify it.
- Draw a digit in the sketchpad tab.
- Show top-3 class probabilities.

