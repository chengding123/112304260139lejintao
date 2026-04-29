# Kaggle Digit Recognizer

## 项目简介

本项目用于完成 Kaggle 比赛 **Digit Recognizer**。任务是根据 `28x28` 灰度手写数字图像预测数字 `0-9`，评价指标为分类准确率 `Accuracy`。

本目录按 `比赛2` 的 GitHub 提交结构整理，保留脚本、报告、指标文件和可直接提交到 Kaggle 的 `submission.csv`。原始数据、外部数据缓存和模型权重不上传。

## 当前结果

- 已知线上反馈最高准确率：`0.99571`
- 已知线上反馈对应版本：`submissions/no_external/submission.csv`
- 当前冲榜激进版本：`submissions/aggressive_external/submission.csv`
- 说明：激进版本使用外部手写数字数据预训练，提交前应确认比赛规则是否允许外部数据。

## 两次结果区分

| 版本 | 是否使用外部数据 | 提交文件 | 说明 |
|---|---|---|---|
| 非外部数据版 | 否 | `submissions/no_external/submission.csv` | 已知线上准确率 `0.99571` |
| 激进外部数据版 | 是 | `submissions/aggressive_external/submission.csv` | 后续冲榜版本，融合外部数据预训练模型 |

## 目录结构

```text
.
├── README.md
├── .gitignore
├── data/
│   └── README.md
├── docs/
│   ├── competition_requirements.md
│   └── reports/
│       ├── aggressive_external_report.md
│       └── no_external_report.md
├── metrics/
│   ├── aggressive_external_metrics.json
│   ├── no_external_metrics.json
│   ├── external_fast_training_metrics.json
│   ├── pseudo_training_metrics.json
│   └── wide_full_training_metrics.json
├── scripts/
│   ├── digit_recognizer_cnn.py
│   ├── digit_recognizer_ensemble.py
│   ├── digit_recognizer_external.py
│   ├── digit_recognizer_pseudo.py
│   └── digit_recognizer_wide_full.py
└── submissions/
    ├── aggressive_external/
    │   └── submission.csv
    └── no_external/
        └── submission.csv
```

## 运行方式

### 1. 训练基础 CNN

```powershell
py scripts\digit_recognizer_cnn.py --full-train --epochs 12 --batch-size 256 --lr 0.0015
```

### 2. 训练宽 CNN

```powershell
py scripts\digit_recognizer_wide_full.py --epochs 10 --batch-size 256 --lr 0.0012
```

### 3. 训练外部数据预训练模型

```powershell
py scripts\digit_recognizer_external.py --external-limit 80000 --pretrain-epochs 1 --finetune-epochs 4
```

## 当前建议

如果按已知线上最高准确率整理，提交：

- `submissions/no_external/submission.csv`

如果继续尝试冲分，提交：

- `submissions/aggressive_external/submission.csv`

如果需要避免外部数据规则风险，也使用：

- `submissions/no_external/submission.csv`
