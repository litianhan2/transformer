import numpy as np  # 导入NumPy数值计算库，用于数组操作和数学计算
import pandas as pd  # 导入Pandas数据分析库，用于数据读取和处理
from sklearn.ensemble import GradientBoostingClassifier  # 导入sklearn梯度提升分类器
from sklearn.model_selection import StratifiedKFold  # 导入sklearn分层K折交叉验证
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score  # 导入sklearn评估指标
import json  # 导入JSON模块，用于保存和加载JSON格式数据
import os  # 导入操作系统模块，用于文件和目录操作
import warnings  # 导入Python警告模块，用于控制警告信息的显示
warnings.filterwarnings('ignore')  # 过滤并忽略所有警告信息，避免输出干扰


def load_data(data_path, mirna_feat_path, drug_feat_path, negative_ratio=1):
    """
    加载并处理miRNA-药物关联数据，生成训练所需的特征和标签
    
    参数:
        data_path: 正样本数据文件路径（miRNA-药物对）
        mirna_feat_path: miRNA特征文件路径（k-mer特征）
        drug_feat_path: 药物特征文件路径（GIN特征）
        negative_ratio: 负样本与正样本的比例，默认为1:1
    
    返回:
        X: 特征矩阵（miRNA特征+药物特征拼接）
        y: 标签向量（1为正样本，0为负样本）
        all_data: 完整数据框
    """
    positive_data = pd.read_csv(data_path, header=None)  # 从CSV文件读取正样本数据，无表头
    positive_data.columns = ['miRNA', 'drug', 'pubchem_id']  # 为数据框设置列名
    positive_data['label'] = 1  # 为所有正样本添加标签列，值为1
    
    mirna_feat = pd.read_csv(mirna_feat_path, header=None)  # 读取miRNA特征数据
    drug_feat = pd.read_csv(drug_feat_path, header=None)  # 读取药物特征数据
    
    all_mirnas = positive_data['miRNA'].unique()  # 获取所有唯一的miRNA名称
    all_drugs = positive_data['drug'].unique()  # 获取所有唯一的药物名称
    
    mirna_to_idx = {m: i for i, m in enumerate(all_mirnas)}  # 创建miRNA名称到索引的映射字典
    drug_to_idx = {d: i for i, d in enumerate(all_drugs)}  # 创建药物名称到索引的映射字典
    
    positive_pairs = set(zip(positive_data['miRNA'], positive_data['drug']))  # 创建正样本对的集合
    
    np.random.seed(42)  # 设置随机种子为42，确保结果可复现
    negative_samples = []  # 初始化负样本列表
    for _ in range(len(positive_data) * negative_ratio):  # 循环生成指定数量的负样本
        while True:  # 持续循环直到生成有效的负样本
            mirna = np.random.choice(all_mirnas)  # 随机选择一个miRNA
            drug = np.random.choice(all_drugs)  # 随机选择一个药物
            if (mirna, drug) not in positive_pairs:  # 如果该miRNA-药物对不在正样本集合中
                negative_samples.append({  # 添加到负样本列表
                    'miRNA': mirna,
                    'drug': drug,
                    'label': 0,
                    'mirna_idx': mirna_to_idx[mirna],
                    'drug_idx': drug_to_idx.get(drug, 0)
                })
                break  # 跳出while循环，继续生成下一个负样本
    
    negative_data = pd.DataFrame(negative_samples)  # 将负样本列表转换为数据框
    
    positive_data['mirna_idx'] = positive_data['miRNA'].map(mirna_to_idx)  # 添加miRNA索引列
    positive_data['drug_idx'] = positive_data['drug'].map(lambda x: drug_to_idx.get(x, 0))  # 添加药物索引列
    
    all_data = pd.concat([positive_data, negative_data], ignore_index=True)  # 合并正负样本数据
    all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)  # 随机打乱数据顺序
    
    X_mirna = mirna_feat.values[all_data['mirna_idx'].values]  # 根据索引获取miRNA特征
    X_drug = drug_feat.values[all_data['drug_idx'].values]  # 根据索引获取药物特征
    X = np.concatenate([X_mirna, X_drug], axis=1)  # 沿列方向拼接miRNA和药物特征
    y = all_data['label'].values  # 获取标签向量
    
    return X, y, all_data  # 返回特征矩阵、标签向量和数据框


def cross_validation(X, y, n_splits=5):
    """
    执行5折分层交叉验证，评估Gradient Boosting模型性能
    
    参数:
        X: 特征矩阵
        y: 标签向量
        n_splits: 交叉验证折数，默认为5
    
    返回:
        results: 包含各折结果和平均结果的字典
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)  # 创建分层K折交叉验证器
    
    fold_results = []  # 初始化折结果列表
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):  # 遍历每一折
        print(f"\n{'='*50}")  # 打印分隔线
        print(f"Fold {fold + 1}/{n_splits}")  # 打印当前折数
        print(f"{'='*50}")  # 打印分隔线
        
        X_train, X_test = X[train_idx], X[test_idx]  # 分割训练集和测试集
        y_train, y_test = y[train_idx], y[test_idx]  # 分割训练标签和测试标签
        
        model = GradientBoostingClassifier(  # 创建梯度提升分类器实例
            n_estimators=200,  # 提升迭代次数（决策树数量）为200
            learning_rate=0.1,  # 学习率为0.1，控制每棵树对最终结果的贡献
            max_depth=5,  # 每棵决策树的最大深度为5
            min_samples_split=5,  # 节点分裂所需的最小样本数
            min_samples_leaf=2,  # 叶子节点所需的最小样本数
            random_state=42  # 随机种子为42，确保结果可复现
        )
        
        model.fit(X_train, y_train)  # 使用训练数据拟合模型
        
        y_pred = model.predict(X_test)  # 对测试集进行预测
        y_prob = model.predict_proba(X_test)[:, 1]  # 获取正类预测概率
        
        accuracy = accuracy_score(y_test, y_pred)  # 计算准确率
        auc = roc_auc_score(y_test, y_prob)  # 计算AUC值
        f1 = f1_score(y_test, y_pred)  # 计算F1分数
        precision = precision_score(y_test, y_pred, zero_division=0)  # 计算精确率
        recall = recall_score(y_test, y_pred, zero_division=0)  # 计算召回率
        
        fold_results.append({  # 记录该折结果
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })
        
        print(f"Accuracy: {accuracy:.4f}")  # 打印准确率
        print(f"AUC: {auc:.4f}")  # 打印AUC值
        print(f"F1-Score: {f1:.4f}")  # 打印F1分数
        print(f"Precision: {precision:.4f}")  # 打印精确率
        print(f"Recall: {recall:.4f}")  # 打印召回率
    
    avg_acc = np.mean([r['accuracy'] for r in fold_results])  # 计算平均准确率
    std_acc = np.std([r['accuracy'] for r in fold_results])  # 计算准确率标准差
    avg_auc = np.mean([r['auc'] for r in fold_results])  # 计算平均AUC
    avg_f1 = np.mean([r['f1'] for r in fold_results])  # 计算平均F1
    avg_precision = np.mean([r['precision'] for r in fold_results])  # 计算平均精确率
    avg_recall = np.mean([r['recall'] for r in fold_results])  # 计算平均召回率
    
    print(f"\n{'='*50}")  # 打印分隔线
    print("Cross-Validation Results (Gradient Boosting):")  # 打印交叉验证结果标题
    print(f"{'='*50}")  # 打印分隔线
    print(f"Average Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")  # 打印平均准确率
    print(f"Average AUC: {avg_auc:.4f}")  # 打印平均AUC
    print(f"Average F1-Score: {avg_f1:.4f}")  # 打印平均F1分数
    print(f"Average Precision: {avg_precision:.4f}")  # 打印平均精确率
    print(f"Average Recall: {avg_recall:.4f}")  # 打印平均召回率
    
    results = {  # 创建结果字典
        'model': 'Gradient Boosting',  # 模型名称
        'avg_accuracy': avg_acc,  # 平均准确率
        'std_accuracy': std_acc,  # 准确率标准差
        'avg_auc': avg_auc,  # 平均AUC
        'avg_f1': avg_f1,  # 平均F1分数
        'avg_precision': avg_precision,  # 平均精确率
        'avg_recall': avg_recall,  # 平均召回率
        'fold_results': fold_results  # 各折结果
    }
    
    return results  # 返回结果字典


if __name__ == '__main__':  # 主程序入口
    os.makedirs('results', exist_ok=True)  # 创建结果保存目录
    
    print("="*60)  # 打印分隔线
    print("Gradient Boosting Model Training")  # 打印模型训练标题
    print("="*60)  # 打印分隔线
    
    print("\nLoading data...")  # 打印加载数据提示
    X, y, data = load_data(  # 加载数据
        'data_3000.csv',  # 正样本数据路径
        'miRNA_kmer.csv',  # miRNA特征路径
        'drug_GIN_64.csv',  # 药物特征路径
        negative_ratio=1  # 负样本比例1:1
    )
    
    print(f"Dataset size: {len(X)}")  # 打印数据集大小
    print(f"Feature dimension: {X.shape[1]}")  # 打印特征维度
    print(f"Positive samples: {sum(y)}")  # 打印正样本数量
    print(f"Negative samples: {len(y) - sum(y)}")  # 打印负样本数量
    
    print("\nStarting 5-fold cross-validation...")  # 打印开始交叉验证提示
    results = cross_validation(X, y, n_splits=5)  # 执行5折交叉验证
    
    with open('results/gb_results.json', 'w') as f:  # 打开结果文件
        json.dump(results, f, indent=2)  # 保存结果到JSON文件
    
    print(f"\nResults saved to results/gb_results.json")  # 打印结果保存提示
    print("\nTraining completed!")  # 打印训练完成提示
