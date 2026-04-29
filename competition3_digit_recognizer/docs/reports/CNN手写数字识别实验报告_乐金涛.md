# 机器学习实验：基于CNN的手写数字识别

## 1. 学生信息

- **姓名**：乐金涛
- **学号**：112304260139
- **班级**：数据1231

---

## 2. 实验概述

本实验使用 Kaggle **Digit Recognizer** 比赛数据完成手写数字识别任务。数据来自 `digit-recognizer.zip`，其中 `train.csv` 包含 `42000` 张带标签的 `28×28` 灰度手写数字图像，`test.csv` 包含 `28000` 张待预测图像。模型目标是将每张图像分类为 `0-9` 中的一个数字，Kaggle 评价指标为 `Accuracy`。

本次实验完成内容：

| 阶段 | 内容 | 完成情况 |
|------|------|----------|
| 实验一 | 模型训练与超参数调优 | 已完成，Kaggle 线上反馈最高准确率 `0.99571` |
| 实验二 | 模型封装与 Web 部署 | 已完成本地 Gradio 项目代码，公网部署链接待发布 |
| 实验三 | 交互式手写识别系统 | 已在 Gradio 代码中加入 Sketchpad/手写输入页，待部署后截图 |

---

## 3. 实验环境

- Python 3.12
- PyTorch 2.11.0
- torchvision 0.26.0
- numpy / pandas / scikit-learn / matplotlib
- Gradio：Web 应用代码已编写，当前本机环境尚未安装运行

---

## 实验一：模型训练与超参数调优（必做）

### 1.1 实验目标

使用 CNN 在 Kaggle Digit Recognizer 数据集上完成手写数字分类，并通过超参数调优达到 Kaggle 评分 `0.98+`。最终非外部数据版本线上反馈准确率为 `0.99571`，达到实验要求。

### 1.2 模型结构（统一）

对比实验使用统一基础 CNN：

```text
输入(1×28×28)
→ Conv2d(1, 32, 3×3) + ReLU + MaxPool
→ Conv2d(32, 64, 3×3) + ReLU + MaxPool
→ Flatten
→ Linear(64×7×7, 128) + ReLU
→ Linear(128, 10)
→ 输出(10类)
```

数据划分方式：

- 训练集：`80%`
- 验证集：`10%`
- 本地测试集：`10%`
- 划分方式：按标签分层划分

### 1.3 超参数对比实验

| 实验编号 | 优化器 | 学习率 | Batch Size | 数据增强 | Early Stopping |
|----------|--------|--------|------------|----------|----------------|
| Exp1 | SGD | 0.01 | 64 | 否 | 否 |
| Exp2 | Adam | 0.001 | 64 | 否 | 否 |
| Exp3 | Adam | 0.001 | 128 | 否 | 是 |
| Exp4 | Adam | 0.001 | 64 | 是 | 是 |

实验结果如下，`Test Acc` 为从 Kaggle 训练集内部划分出的本地测试集准确率：

| 实验编号 | Train Acc | Val Acc | Test Acc | 最低 Loss | 收敛 Epoch |
|----------|-----------|---------|----------|-----------|------------|
| Exp1 | 99.86% | 98.93% | 98.93% | 0.0432 | 9 |
| Exp2 | 99.57% | 98.81% | 98.48% | 0.0473 | 4 |
| Exp3 | 99.77% | 98.90% | 98.95% | 0.0455 | 8 |
| Exp4 | 99.36% | 99.19% | 99.17% | 0.0310 | 9 |

结论：Exp4 使用 Adam、数据增强和 Early Stopping，在验证集和本地测试集上表现最好，泛化能力最强。

### 1.4 最终提交模型

最终用于 Kaggle 提交的非外部数据版本不是单个基础 CNN，而是多个 CNN 模型的加权融合。该版本对应文件为：

- `submission_candidate_wideblend.csv`
- GitHub 路径：`competition3_digit_recognizer/submissions/no_external/submission.csv`

| 配置项 | 你的设置 |
|--------|----------|
| 优化器 | AdamW |
| 学习率 | 主要为 `0.0012-0.0015` |
| Batch Size | 256 |
| 训练 Epoch 数 | 10-14，不同随机种子模型略有不同 |
| 是否使用数据增强 | 是 |
| 数据增强方式（如有） | 平移、轻微旋转、仿射变换、噪声扰动、TTA |
| 是否使用 Early Stopping | 对 KFold/验证实验使用；全量训练模型不使用 |
| 是否使用学习率调度器 | 是，OneCycleLR |
| 其他调整（如有） | 多随机种子全量训练、宽 CNN、KFold 权重、伪标签、高置信预测融合 |
| **Kaggle Score** | `0.99571` |

另外保留了一个激进版本：

- `submission_aggressive_final.csv`
- GitHub 路径：`competition3_digit_recognizer/submissions/aggressive_external/submission.csv`
- 说明：该版本使用外部手写数字数据预训练模型，需确认比赛规则允许后再正式提交。

### 1.5 Loss 曲线

Loss 曲线已使用 `matplotlib` 绘制，文件位置：

![Loss 曲线](template_outputs/loss_curve.png)

原始输出文件：

- `template_outputs/loss_curve.png`
- `template_outputs/template_experiment_results.json`

### 1.6 分析问题

**Q1：Adam 和 SGD 的收敛速度有何差异？从实验结果中你观察到了什么？**

Adam 的前期收敛速度明显更快。Exp2 在第 4 个 epoch 达到最佳验证准确率 `98.81%`，而 SGD 的 Exp1 在第 9 个 epoch 才达到最佳验证准确率 `98.93%`。SGD 加 momentum 在训练后期也能达到较高准确率，但需要更多 epoch。

**Q2：学习率对训练稳定性有什么影响？**

学习率过大会导致验证 Loss 波动，学习率过小会导致收敛慢。本实验中 `0.001` 的 Adam 训练稳定，能够较快达到 `98%+`；最终模型使用 AdamW 配合 OneCycleLR，让学习率先升后降，后期收敛更平滑。

**Q3：Batch Size 对模型泛化能力有什么影响？**

Batch Size 从 64 增加到 128 后，训练更平稳，但单次更新次数减少。Exp3 的本地测试准确率 `98.95%` 略高于 Exp2 的 `98.48%`，说明较大的 batch 在本实验中并未损害泛化，但最佳结果仍来自 Batch Size 64 加数据增强的 Exp4。

**Q4：Early Stopping 是否有效防止了过拟合？**

有效。Exp3 和 Exp4 在验证集不再提升后提前停止，避免继续训练造成验证 Loss 回升。Exp4 在第 9 个 epoch 达到最佳验证准确率，之后验证 Loss 开始波动，Early Stopping 有助于保留泛化更好的模型。

**Q5：数据增强是否提升了模型的泛化能力？为什么？**

提升明显。Exp4 使用随机旋转和平移仿射增强后，验证准确率达到 `99.19%`，本地测试准确率达到 `99.17%`，均高于其他对比实验。原因是手写数字存在位置、倾斜角度、书写粗细差异，数据增强能让模型学习更稳定的数字形状特征，而不是记住训练集中的固定像素位置。

### 1.7 提交清单

- [x] 对比实验结果表格（1.3）
- [x] 最终模型超参数配置（1.4）
- [x] Loss 曲线图（1.5）
- [x] 分析问题回答（1.6）
- [x] Kaggle 预测结果 CSV
- [ ] Kaggle Score 截图（需在 Kaggle 提交页面截图后补入）

---

## 实验二：模型封装与 Web 部署（必做）

### 2.1 实验目标

将实验一训练好的 CNN 模型封装为 Web 服务，实现上传图片、模型预测、输出结果的完整流程。

### 2.2 技术方案

已创建 Gradio Web 应用目录：

```text
cnn_digit_web_app/
├── app.py
├── model.pth
├── requirements.txt
└── README.md
```

功能包括：

1. 上传手写数字图片
2. 自动转为灰度 `28×28` 图像
3. 根据 MNIST/Kaggle 格式自动处理黑白背景
4. 加载 PyTorch 模型并预测数字类别
5. 显示 Top-3 概率

本地运行命令：

```powershell
cd cnn_digit_web_app
py -m pip install -r requirements.txt
py app.py
```

### 2.3 项目结构

```text
cnn_digit_web_app/
├── app.py              # Web 应用入口
├── model.pth           # 训练好的模型权重
├── requirements.txt    # 依赖列表
└── README.md           # 项目说明
```

### 2.4 部署要求

建议部署平台：HuggingFace Spaces。

部署步骤：

1. 新建 HuggingFace Space，SDK 选择 Gradio
2. 上传 `cnn_digit_web_app/` 目录下的文件
3. Space 自动安装依赖并运行 `app.py`
4. 获得公网访问链接

### 2.5 提交信息

| 提交项 | 内容 |
|--------|------|
| GitHub 仓库地址 | `https://github.com/chengding123/112304260139lejintao/tree/main/competition3_digit_recognizer` |
| 在线访问链接 | 待部署到 HuggingFace Spaces 后填写 |

截图项：

- Web 页面截图：待部署后补入
- 预测结果截图：待部署后补入

### 2.6 提交清单

- [x] GitHub 仓库地址
- [ ] 在线访问链接（部署后补入）
- [ ] 页面截图与预测结果截图（部署后补入）

---

## 实验三：交互式手写识别系统（选做，加分）

### 3.1 实验目标

在实验二的基础上，将上传图片升级为网页手写板输入，实现用户直接手写数字并识别。

### 3.2 功能实现

`cnn_digit_web_app/app.py` 已加入 Gradio `ImageEditor` 手写输入页：

| 功能 | 实现情况 |
|------|----------|
| 手写输入 | 已在 Sketchpad 标签页实现 |
| 实时识别 | 用户修改图片后触发预测 |
| 连续使用 | Gradio 组件支持清空和重复输入 |

### 3.3 加分项

| 加分项 | 实现情况 |
|--------|----------|
| 显示 Top-3 预测结果及置信度 | 已实现 |
| 显示概率分布 | 已通过 `gr.Label(num_top_classes=3)` 展示 |
| 历史识别记录展示 | 未实现 |

### 3.4 提交信息

| 提交项 | 内容 |
|--------|------|
| 在线访问链接 | 待部署到 HuggingFace Spaces 后填写 |
| 实现了哪些加分项 | Top-3 预测结果及置信度、概率显示 |

截图项：

- 手写输入截图：待部署后补入
- 识别结果截图：待部署后补入

### 3.5 提交清单

- [ ] 在线系统链接（部署后补入）
- [ ] 手写输入与识别结果截图（部署后补入）

---

## 评分标准对应说明

| 项目 | 分值 | 完成说明 |
|------|------|----------|
| 实验一：模型训练与调优 | 60 分 | 已完成 4 组对比实验、Loss 曲线、分析问题，Kaggle 线上分数 `0.99571` |
| 实验二：Web 部署 | 30 分 | 已完成 Gradio 项目代码和本地运行文件，公网部署链接待补 |
| 实验三：交互系统（加分） | 10 分 | 已实现手写输入代码和 Top-3 概率显示，部署截图待补 |
| **总计** | **100 分** | 训练部分已完整，部署截图需后续补齐 |

