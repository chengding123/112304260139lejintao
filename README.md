# Kaggle Word2Vec NLP Tutorial 实验说明

## 1. 项目目标

本项目针对 Kaggle 比赛 **Bag of Words Meets Bags of Popcorn** 进行电影评论情感分类。  
比赛评测指标为 **AUC（ROC 曲线下面积）**，因此所有实验都以提升 AUC 为目标，而不是追求 Accuracy。

## 2. 当前最高分方案

当前本地验证表现最好的方案是：

- **5 折 OOF 融合**
- 融合模型包括：
  - `char TF-IDF + Logistic Regression`
  - `word TF-IDF + LinearSVC`
  - `NB-SVM`
- 使用无标签数据 `unlabeledTrainData.tsv.zip` 参与词表拟合
- 最终本地 **OOF AUC = 0.972237**

### 对应代码

- 主代码：`train_oof_submission.py`
- 实验报告：`submission_oof_report.md`
- 对应提交文件：`submission_oof.csv`

## 3. 推荐提交文件

当前优先推荐上传：

- `submission_oof.csv`

原因：

- 它是当前所有已完成实验中本地 AUC 最高的一版
- 它采用 OOF 融合，通常比单次切分验证更稳

## 4. 实验结果汇总

### 实验 1：基础版 TF-IDF + Logistic Regression

- 代码：`train_auc_submission.py`
- 报告：`submission_report.md`
- 提交文件：`submission.csv`
- 验证集 AUC：**0.968304**

说明：

- 对文本做基础清洗
- 使用词级 TF-IDF（1-2 gram）
- 用 Logistic Regression 输出正类概率

### 实验 2：词级 + 字符级 + SGD 融合

- 代码：`train_highscore_submission.py`
- 报告：`submission_highscore_report.md`
- 提交文件：`submission_highscore.csv`
- 融合后验证 AUC：**0.968918**

说明：

- 引入字符级 TF-IDF
- 同时使用词级 LR、字符级 LR、词级 SGD
- 按排序分数做加权融合

### 实验 3：Ultra 融合版

- 代码：`train_ultra_submission.py`
- 报告：`submission_ultra_report.md`
- 提交文件：`submission_ultra.csv`
- 最优融合验证 AUC：**0.972215**

说明：

- 引入 `LinearSVC`
- 引入 `NB-SVM`
- 使用无标签数据参与向量器词表构建
- 自动搜索融合权重

### 实验 4：OOF 多模型融合版（当前最优）

- 代码：`train_oof_submission.py`
- 报告：`submission_oof_report.md`
- 提交文件：`submission_oof.csv`
- **5 折 OOF 最优融合 AUC：0.972237**
- 折内融合 AUC 均值：**0.972189**

说明：

- 使用 5 折交叉验证获得 OOF 预测
- 模型包含 `char_lr + word_svc + nbsvm`
- 最优融合权重为：
  - `char_lr = 0.20`
  - `word_svc = 0.35`
  - `nbsvm = 0.45`

## 5. 文件说明

### 数据文件

- `word2vec-nlp-tutorial.zip`：比赛总压缩包
- `labeledTrainData.tsv` / `labeledTrainData.tsv.zip`：带标签训练集
- `sampleSubmission.csv`：官方提交模板，已包含在比赛压缩包中

### 处理与实验脚本

- `preprocess_labeled_train.py`：基础清洗与训练/验证集切分
- `train_auc_submission.py`：基础 TF-IDF + LR
- `train_highscore_submission.py`：词/字符/SGD 融合
- `train_ultra_submission.py`：Ultra 融合实验
- `train_oof_submission.py`：当前最高分方案

### 输出结果

- `submission.csv`
- `submission_highscore.csv`
- `submission_ultra.csv`
- `submission_oof.csv`

其中建议最终提交：

- `submission_oof.csv`

## 6. 复现方式

在当前目录下运行：

```bash
py train_oof_submission.py
```

运行后会生成：

- `submission_oof.csv`
- `submission_oof_report.md`

## 7. 方法总结

从当前实验结果看：

- 单一 TF-IDF + LR 已经能得到较强基线
- 字符级特征对提升 AUC 有帮助
- `LinearSVC` 和 `NB-SVM` 对这个任务很有效
- 在这个比赛上，**融合** 比单模型更重要
- **OOF 融合** 比单次切分验证更稳，因此当前作为最终推荐方案

## 8. 当前结论

如果现在要提交比赛结果，请优先上传：

- `submission_oof.csv`

如果要继续优化，可以继续尝试：

- 多随机种子 OOF 融合
- Word2Vec / Doc2Vec / FastText 类型的语义向量方法
- 更细致的 stacking 或 blending
