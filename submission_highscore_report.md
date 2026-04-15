# High-score AUC 提交说明

- 词级 LR 验证 AUC: 0.967796
- 字符级 LR 验证 AUC: 0.960697
- 词级 SGD 验证 AUC: 0.963465
- 融合后验证 AUC: 0.968918
- 融合权重: word_lr=0.45, char_lr=0.35, word_sgd=0.20
- 提交文件: submission_highscore.csv
- 注意: 为适配 AUC，提交值使用融合后的排序分数，而不是硬标签。