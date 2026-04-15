# Kaggle Word2Vec NLP Tutorial

## 项目简介

本项目用于完成 Kaggle 比赛 **Bag of Words Meets Bags of Popcorn** 的电影评论情感分类任务。  
比赛评价指标是 **AUC（ROC 曲线下面积）**，因此所有实验都以 AUC 为核心优化目标。

## 当前最优结果

- 最优方案：`scripts/train_oof_submission.py`
- 方法：5 折 OOF 融合
- 融合模型：`char TF-IDF + Logistic Regression`、`word TF-IDF + LinearSVC`、`NB-SVM`
- 本地 OOF AUC：**0.972237**
- 推荐提交文件：`submissions/submission_oof.csv`
- 对应实验报告：`docs/reports/submission_oof_report.md`

## 目录结构

```text
.
├─ README.md
├─ .gitignore
├─ data/
│  ├─ raw/                 # 本地原始数据，默认不上传
│  └─ processed/           # 本地清洗结果，默认不上传
├─ docs/
│  ├─ competition_requirements.md
│  └─ reports/             # 各实验报告
├─ scripts/                # 训练与预处理脚本
└─ submissions/            # 可直接上传 Kaggle 的提交文件
```

## 实验结果汇总

| 实验 | 脚本 | 本地 AUC | 提交文件 |
|---|---|---:|---|
| 基础 TF-IDF + LR | `scripts/train_auc_submission.py` | 0.968304 | `submissions/submission.csv` |
| 词/字符/SGD 融合 | `scripts/train_highscore_submission.py` | 0.968918 | `submissions/submission_highscore.csv` |
| Ultra 融合版 | `scripts/train_ultra_submission.py` | 0.972215 | `submissions/submission_ultra.csv` |
| 5 折 OOF 融合 | `scripts/train_oof_submission.py` | **0.972237** | `submissions/submission_oof.csv` |

详细报告见：

- `docs/reports/submission_report.md`
- `docs/reports/submission_highscore_report.md`
- `docs/reports/submission_ultra_report.md`
- `docs/reports/submission_oof_report.md`

## 运行方式

### 1. 预处理数据

```bash
py scripts/preprocess_labeled_train.py
```

### 2. 运行当前最优方案

```bash
py scripts/train_oof_submission.py
```

运行后会生成：

- `submissions/submission_oof.csv`
- `docs/reports/submission_oof_report.md`

## 文件说明

### 文档

- `docs/competition_requirements.md`：比赛任务与数据说明
- `docs/reports/`：不同实验方案的结果报告

### 脚本

- `scripts/preprocess_labeled_train.py`：基础清洗与训练/验证划分
- `scripts/train_auc_submission.py`：基础 TF-IDF + LR
- `scripts/train_highscore_submission.py`：词/字符/SGD 融合
- `scripts/train_ultra_submission.py`：Ultra 融合实验
- `scripts/train_oof_submission.py`：当前最优方案
- `scripts/train_multiseed_oof_submission.py`：多随机种子 OOF 融合实验
- `scripts/train_word2vec_avg_submission.py`：Word2Vec 均值向量实验

### 提交结果

- `submissions/submission.csv`
- `submissions/submission_highscore.csv`
- `submissions/submission_ultra.csv`
- `submissions/submission_oof.csv`

## 当前建议

如果现在需要上传比赛结果，优先提交：

- `submissions/submission_oof.csv`

如果要继续优化，可继续尝试：

- 多随机种子 OOF 融合
- Word2Vec / FastText / Doc2Vec 类型的语义特征
- 更细的 stacking 或 blending
