# OOF AUC 提交说明

- 5 折 OOF 词级 LR AUC: 0.966962
- 5 折 OOF 字符级 LR AUC: 0.960672
- 5 折 OOF 词级 LinearSVC AUC: 0.968476
- 5 折 OOF NB-SVM AUC: 0.967753
- 5 折 OOF 最优融合 AUC: 0.972237
- 折内融合 AUC 均值: 0.972189
- 最优融合权重: char_lr=0.20, word_svc=0.35, nbsvm=0.45
- 无标签数据用途: 参与向量器词表拟合，但不参与监督训练。
- 提交文件: submission_oof.csv
- 注意: 为适配 AUC，提交值使用融合后的排序分数，而不是硬标签。