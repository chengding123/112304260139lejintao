# No External Report

## Summary

This version avoids external training data and keeps the safer competition-data-only path separate from the aggressive route.

- Submission: `submissions/no_external/submission.csv`
- Metrics file: `metrics/no_external_metrics.json`
- Method: weighted probability blend
- External data: no
- Reported online accuracy: `0.99571`

## Model Blend

The blend uses:

- full-data CNN models trained with several random seeds
- pseudo-label CNN model based on model agreement over the test set
- full-data wide CNN model
- KFold wide CNN checkpoints

The exact blend weights are recorded in `metrics/no_external_metrics.json`.

## Submission Check

The submission file was verified to contain:

- `28000` rows
- columns `ImageId,Label`
- labels from `0` to `9`
- no missing values

## Note

This was the main non-external candidate before adding external-data-pretrained models. It is kept separately so the two competition attempts are distinguishable in GitHub.
