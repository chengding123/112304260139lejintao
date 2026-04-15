# AUC 验证与提交文件说明

- 训练集样本数: 25000
- 测试集样本数: 25000
- 验证集 AUC: 0.968304
- 模型: TF-IDF (1-2 gram) + Logistic Regression
- 提交文件: submission.csv
- 提交列说明: `id`, `sentiment`
- 注意: 为匹配 AUC 指标，`sentiment` 输出的是正类概率，而不是硬分类 0/1。