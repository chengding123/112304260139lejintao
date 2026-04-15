# Ultra AUC 提交说明

- 词级 LR 验证 AUC: 0.967969
- 字符级 LR 验证 AUC: 0.960658
- 词级 LinearSVC 验证 AUC: 0.969270
- NB-SVM 验证 AUC: 0.967250
- 最优融合验证 AUC: 0.972215
- 最优融合权重: char_lr=0.15, word_svc=0.45, nbsvm=0.40
- 无标签数据用途: 参与向量器词表拟合，但不参与监督训练。
- 提交文件: submission_ultra.csv
- 注意: 为适配 AUC，提交值使用融合后的排序分数，而不是硬标签。