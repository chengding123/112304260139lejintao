# Kaggle 比赛要求整理

比赛名称：Bag of Words Meets Bags of Popcorn

## 任务目标

- 对 IMDB 电影评论做情感二分类。
- `sentiment=1` 表示正向评论，`sentiment=0` 表示负向评论。
- 最终提交时，需要针对测试集生成每条评论的情感预测结果。

## 评测方式

- 比赛使用 ROC 曲线下面积（AUC, Area Under the ROC Curve）作为评分指标。

## 数据组成

- `labeledTrainData.tsv`: 带标签训练集，包含 `id / sentiment / review` 三列。
- `testData.tsv`: 测试集，通常不含 `sentiment` 标签。
- `unlabeledTrainData.tsv`: 无标签评论，可用于无监督词向量或半监督方法。

## 我在当前数据里核对到的情况

- `labeledTrainData.tsv.zip` 解压后包含 `labeledTrainData.tsv`。
- 当前训练集共有 25,000 条样本。
- 标签分布均衡：正样本 12,500 条，负样本 12,500 条。
- 字段完整，无缺失值，`id` 无重复。

## 已完成的预处理

- 解压 `labeledTrainData.tsv.zip`
- 去除评论中的 HTML 标签
- 仅保留英文字母字符
- 全部转为小写
- 压缩多余空格
- 生成清洗后的文本列及基础长度特征
- 按标签分层切分出训练集/验证集

## 参考来源

- Kaggle 竞赛页: https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview
- 项目报告整理页: https://people.inf.elte.hu/csabix/publications/other/ML_ProjectReport.pdf
- 比赛代码解读页: https://uumini.tistory.com/74
