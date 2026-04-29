# Aggressive External Report

## Summary

This is the aggressive leaderboard-pushing version currently prepared for Competition 3.

- Submission: `submissions/aggressive_external/submission.csv`
- Metrics file: `metrics/aggressive_external_metrics.json`
- Method: weighted probability blend
- External data: yes
- Online score: not verified in this workspace

## Model Blend

The final blend uses:

- full-data CNN models trained with different random seeds
- pseudo-label CNN model
- full-data wide CNN model
- two external-data-pretrained wide CNN models
- two previous KFold wide CNN checkpoints

The exact blend weights are recorded in `metrics/aggressive_external_metrics.json`.

## Submission Check

The submission file was verified to contain:

- `28000` rows
- columns `ImageId,Label`
- labels from `0` to `9`
- no missing values

## Risk

This route uses external handwritten digit data for pretraining. It is separated from the non-external version because it may carry competition-rule risk depending on the active Kaggle rules.
