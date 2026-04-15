# Data Directory

本目录用于存放本地数据文件：

- `raw/`：Kaggle 原始数据压缩包与解压文件
- `processed/`：预处理后的中间结果

这些文件体积较大，且通常受比赛数据使用规则约束，因此默认不上传到 GitHub。  
如果需要复现实验，请先将比赛数据放入：

- `data/raw/word2vec-nlp-tutorial.zip`
- `data/raw/labeledTrainData.tsv.zip`
