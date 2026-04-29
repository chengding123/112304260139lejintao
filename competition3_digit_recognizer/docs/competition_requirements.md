# Competition Requirements

## Competition

- Name: Kaggle Digit Recognizer
- Task: classify grayscale handwritten digit images into labels `0-9`
- Metric: Accuracy
- Submission columns: `ImageId`, `Label`
- Test rows: `28000`

## Data Format

`train.csv` contains one label column and `784` pixel columns.

`test.csv` contains only the `784` pixel columns.

Each row represents a `28x28` grayscale image.

## Submitted Files

This project keeps two clearly separated submission versions:

- `submissions/aggressive_external/submission.csv`
- `submissions/no_external/submission.csv`

Both files were checked for:

- `28000` rows
- columns exactly equal to `ImageId,Label`
- `ImageId` from `1` to `28000`
- labels in range `0-9`
- no missing labels

## Rule Note

The aggressive version uses external handwritten digit data for pretraining. Before using it as a formal competition submission, verify that the current competition rules allow external data. The non-external version is retained separately for a lower-risk submission path.

