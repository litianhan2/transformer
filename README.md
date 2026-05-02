# miRNA-Drug Resistance Prediction Model

基于Transformer和图神经网络的miRNA-药物耐药性预测模型。

## 项目简介

本项目提出了一种基于深度学习的miRNA-药物耐药性预测方法，结合Transformer编码器和图神经网络(GNN)分别对miRNA序列和药物分子进行特征提取，并通过双线性融合机制实现miRNA与药物特征的交互建模，最终预测miRNA介导的药物耐药性。

## 模型架构

### 核心模型 (Transformer-GCN)

- **miRNA编码器**: 基于Transformer架构，通过多头自注意力机制捕捉miRNA序列中的长程依赖关系
- **药物编码器**: 基于图卷积网络(GCN)，从药物分子图中提取结构特征
- **双线性融合**: 通过双线性交互层建模miRNA与药物之间的复杂关联
- **分类器**: 多层感知机进行最终的耐药性预测

### 集成模型 (Ensemble)

- **Transformer编码器**: 捕捉miRNA序列的全局依赖关系
- **CNN编码器**: 提取miRNA序列的局部特征模式
- **LSTM编码器**: 建模miRNA序列的时序依赖关系
- **注意力融合**: 自适应地融合三种编码器的输出

## 文件说明

| 文件 | 说明 |
|------|------|
| `model.py` | 核心模型定义（Transformer-GCN双线性融合模型） |
| `data_processor.py` | 数据预处理模块（数据集类、负样本生成、分子图构建） |
| `train.py` | 核心模型训练与5折交叉验证 |
| `train_ensemble.py` | 集成模型训练与5折交叉验证 |
| `train_logistic_regression.py` | 逻辑回归基线模型 |
| `train_random_forest.py` | 随机森林基线模型 |
| `train_svm.py` | 支持向量机基线模型 |
| `train_gradient_boosting.py` | 梯度提升基线模型 |

## 依赖环境

- Python 3.6+
- PyTorch
- scikit-learn
- NumPy
- Pandas
- RDKit
- Matplotlib

## 使用方法

### 训练核心模型

```bash
python train.py
```

### 训练集成模型

```bash
python train_ensemble.py
```

### 训练基线模型

```bash
python train_logistic_regression.py
python train_random_forest.py
python train_svm.py
python train_gradient_boosting.py
```

## 评估指标

- Accuracy（准确率）
- AUC（ROC曲线下面积）
- F1-Score（F1分数）
- Precision（精确率）
- Recall（召回率）

所有模型均采用5折分层交叉验证进行评估。
