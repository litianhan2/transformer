import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.optim as optim  # 导入PyTorch优化器模块
from torch.utils.data import DataLoader  # 从PyTorch导入DataLoader，用于批量加载数据
import numpy as np  # 导入NumPy数值计算库
import os  # 导入操作系统模块，用于文件和目录操作
import json  # 导入JSON模块，用于保存和加载JSON格式数据
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score  # 从sklearn导入评估指标
from sklearn.model_selection import StratifiedKFold  # 从sklearn导入分层K折交叉验证
import matplotlib.pyplot as plt  # 导入matplotlib绘图库
from model import miRNADrugResistanceModel  # 从model模块导入miRNA-药物耐药性预测模型
from data_processor import BalancedmiRNADrugDataset, collate_fn  # 从data_processor导入数据集类和批处理函数
import warnings  # 导入警告模块
warnings.filterwarnings('ignore')  # 过滤并忽略所有警告信息


def set_seed(seed=42):  # 定义设置随机种子的函数，确保实验可复现
    np.random.seed(seed)  # 设置NumPy随机种子
    torch.manual_seed(seed)  # 设置PyTorch CPU随机种子
    if torch.cuda.is_available():  # 如果CUDA可用
        torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子


def train_epoch(model, dataloader, optimizer, criterion, device):  # 定义训练一个epoch的函数
    model.train()  # 将模型设置为训练模式
    total_loss = 0  # 初始化总损失为0
    all_preds = []  # 初始化所有预测结果列表
    all_labels = []  # 初始化所有真实标签列表
    
    for batch in dataloader:  # 遍历数据加载器中的每个批次
        mirna_seqs = batch['mirna_seqs'].to(device)  # 将miRNA序列移到指定设备
        labels = batch['labels'].squeeze(-1).to(device)  # 将标签移到指定设备并去除多余维度
        
        optimizer.zero_grad()  # 清零优化器的梯度
        
        batch_outputs = []  # 初始化批次输出列表
        for i, (node_features, adj) in enumerate(batch['drug_data']):  # 遍历批次中的每个药物数据
            node_features = node_features.to(device)  # 将节点特征移到指定设备
            adj = adj.to(device)  # 将邻接矩阵移到指定设备
            output = model(mirna_seqs[i:i+1], node_features, adj)  # 对单个样本进行前向传播
            batch_outputs.append(output)  # 将输出添加到列表中
        
        outputs = torch.cat(batch_outputs)  # 将所有输出拼接成一个张量
        loss = criterion(outputs, labels)  # 计算损失
        
        loss.backward()  # 反向传播计算梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
        optimizer.step()  # 更新模型参数
        
        total_loss += loss.item()  # 累加损失值
        
        preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()  # 将输出转换为预测标签
        all_preds.extend(preds)  # 将预测结果添加到列表
        all_labels.extend(labels.cpu().numpy())  # 将真实标签添加到列表
    
    avg_loss = total_loss / len(dataloader)  # 计算平均损失
    accuracy = accuracy_score(all_labels, all_preds)  # 计算准确率
    
    return avg_loss, accuracy  # 返回平均损失和准确率


def evaluate(model, dataloader, criterion, device):  # 定义评估函数
    model.eval()  # 将模型设置为评估模式
    total_loss = 0  # 初始化总损失为0
    all_preds = []  # 初始化所有预测结果列表
    all_labels = []  # 初始化所有真实标签列表
    all_probs = []  # 初始化所有预测概率列表
    
    with torch.no_grad():  # 禁用梯度计算，节省内存
        for batch in dataloader:  # 遍历数据加载器中的每个批次
            mirna_seqs = batch['mirna_seqs'].to(device)  # 将miRNA序列移到指定设备
            labels = batch['labels'].squeeze(-1).to(device)  # 将标签移到指定设备
            
            batch_outputs = []  # 初始化批次输出列表
            for i, (node_features, adj) in enumerate(batch['drug_data']):  # 遍历批次中的每个药物数据
                node_features = node_features.to(device)  # 将节点特征移到指定设备
                adj = adj.to(device)  # 将邻接矩阵移到指定设备
                output = model(mirna_seqs[i:i+1], node_features, adj)  # 对单个样本进行前向传播
                batch_outputs.append(output)  # 将输出添加到列表中
            
            outputs = torch.cat(batch_outputs)  # 将所有输出拼接成一个张量
            loss = criterion(outputs, labels)  # 计算损失
            
            total_loss += loss.item()  # 累加损失值
            
            probs = torch.sigmoid(outputs).cpu().numpy()  # 计算预测概率
            preds = (probs > 0.5).astype(float)  # 将概率转换为预测标签
            all_preds.extend(preds)  # 将预测结果添加到列表
            all_labels.extend(labels.cpu().numpy())  # 将真实标签添加到列表
            all_probs.extend(probs)  # 将预测概率添加到列表
    
    avg_loss = total_loss / len(dataloader)  # 计算平均损失
    accuracy = accuracy_score(all_labels, all_preds)  # 计算准确率
    auc = roc_auc_score(all_labels, all_probs)  # 计算AUC值
    f1 = f1_score(all_labels, all_preds)  # 计算F1分数
    precision = precision_score(all_labels, all_preds, zero_division=0)  # 计算精确率
    recall = recall_score(all_labels, all_preds, zero_division=0)  # 计算召回率
    
    return avg_loss, accuracy, auc, f1, precision, recall  # 返回所有评估指标


def train_model(data_path, mirna_seq_path, drug_smiles_path,  # 定义模型训练函数
                epochs=200, batch_size=32, lr=0.0005,  # 参数：训练轮数、批次大小、学习率
                negative_ratio=1, device='cuda', save_dir='results'):  # 参数：负样本比例、设备、保存目录
    
    set_seed(42)  # 设置随机种子
    
    os.makedirs(save_dir, exist_ok=True)  # 创建保存目录
    
    train_dataset = BalancedmiRNADrugDataset(  # 创建训练数据集
        data_path, mirna_seq_path, drug_smiles_path,
        negative_ratio=negative_ratio, is_train=True
    )
    
    test_dataset = BalancedmiRNADrugDataset(  # 创建测试数据集
        data_path, mirna_seq_path, drug_smiles_path,
        negative_ratio=negative_ratio, is_train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,  # 创建训练数据加载器
                             shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,  # 创建测试数据加载器
                            shuffle=False, collate_fn=collate_fn)
    
    print(f"训练集大小: {len(train_dataset)}")  # 打印训练集大小
    print(f"测试集大小: {len(test_dataset)}")  # 打印测试集大小
    
    model = miRNADrugResistanceModel(  # 创建模型实例
        mirna_vocab_size=5,  # miRNA词表大小
        mirna_d_model=128,  # miRNA编码器模型维度
        mirna_nhead=8,  # miRNA编码器注意力头数
        mirna_num_layers=4,  # miRNA编码器层数
        mirna_dim_ff=256,  # miRNA编码器前馈网络维度
        drug_node_features=29,  # 药物节点特征维度
        drug_hidden_dim=128,  # 药物编码器隐藏维度
        drug_num_layers=3,  # 药物编码器层数
        fusion_dim=256,  # 融合层维度
        dropout=0.2  # dropout概率
    ).to(device)  # 将模型移到指定设备
    
    criterion = nn.BCEWithLogitsLoss()  # 定义二元交叉熵损失函数（带logits）
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # 定义AdamW优化器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)  # 定义余弦退火学习率调度器
    
    best_accuracy = 0  # 初始化最佳准确率
    best_auc = 0  # 初始化最佳AUC
    history = {  # 初始化训练历史字典
        'train_loss': [], 'train_acc': [],  # 训练损失和准确率
        'test_loss': [], 'test_acc': [], 'test_auc': [], 'test_f1': []  # 测试指标
    }
    
    print("\n开始训练...")  # 打印开始训练信息
    for epoch in range(epochs):  # 遍历每个训练轮次
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)  # 训练一个epoch
        test_loss, test_acc, test_auc, test_f1, test_precision, test_recall = evaluate(  # 评估模型
            model, test_loader, criterion, device
        )
        
        scheduler.step()  # 更新学习率
        
        history['train_loss'].append(train_loss)  # 记录训练损失
        history['train_acc'].append(train_acc)  # 记录训练准确率
        history['test_loss'].append(test_loss)  # 记录测试损失
        history['test_acc'].append(test_acc)  # 记录测试准确率
        history['test_auc'].append(test_auc)  # 记录测试AUC
        history['test_f1'].append(test_f1)  # 记录测试F1
        
        if test_acc > best_accuracy:  # 如果当前测试准确率超过最佳准确率
            best_accuracy = test_acc  # 更新最佳准确率
            best_auc = test_auc  # 更新最佳AUC
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))  # 保存最佳模型
        
        if (epoch + 1) % 5 == 0:  # 每5个epoch打印一次信息
            print(f"Epoch [{epoch+1}/{epochs}]")  # 打印当前epoch
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")  # 打印训练指标
            print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, "  # 打印测试指标
                  f"AUC: {test_auc:.4f}, F1: {test_f1:.4f}")
    
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:  # 保存训练历史到JSON文件
        json.dump(history, f, indent=2)
    
    print(f"\n训练完成!")  # 打印训练完成信息
    print(f"最佳测试准确率: {best_accuracy:.4f}")  # 打印最佳测试准确率
    print(f"最佳测试AUC: {best_auc:.4f}")  # 打印最佳测试AUC
    
    return model, history  # 返回训练好的模型和训练历史


def cross_validation(data_path, mirna_seq_path, drug_smiles_path,  # 定义交叉验证函数
                    n_splits=5, epochs=150, batch_size=32, lr=0.0005,  # 参数：折数、轮数、批次大小、学习率
                    negative_ratio=1, device='cuda'):  # 参数：负样本比例、设备
    
    set_seed(42)  # 设置随机种子
    
    dataset = BalancedmiRNADrugDataset(  # 创建完整数据集
        data_path, mirna_seq_path, drug_smiles_path,
        negative_ratio=negative_ratio, is_train=True
    )
    
    all_data = dataset.data.copy()  # 复制数据
    labels = all_data['label'].values  # 获取标签
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)  # 创建分层K折交叉验证器
    
    fold_results = []  # 初始化折结果列表
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):  # 遍历每一折
        print(f"\n{'='*50}")  # 打印分隔线
        print(f"Fold {fold + 1}/{n_splits}")  # 打印当前折数
        print(f"{'='*50}")  # 打印分隔线
        
        train_dataset = torch.utils.data.Subset(dataset, train_idx)  # 创建训练子集
        test_dataset = torch.utils.data.Subset(dataset, test_idx)  # 创建测试子集
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size,  # 创建训练数据加载器
                                 shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,  # 创建测试数据加载器
                                shuffle=False, collate_fn=collate_fn)
        
        model = miRNADrugResistanceModel(  # 创建模型实例
            mirna_vocab_size=5,
            mirna_d_model=128,
            mirna_nhead=8,
            mirna_num_layers=4,
            mirna_dim_ff=256,
            drug_node_features=29,
            drug_hidden_dim=128,
            drug_num_layers=3,
            fusion_dim=256,
            dropout=0.2
        ).to(device)
        
        criterion = nn.BCEWithLogitsLoss()  # 定义损失函数
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # 定义优化器
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)  # 定义学习率调度器
        
        best_fold_acc = 0  # 初始化该折最佳准确率
        best_fold_auc = 0  # 初始化该折最佳AUC
        
        for epoch in range(epochs):  # 遍历每个训练轮次
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)  # 训练
            test_loss, test_acc, test_auc, test_f1, _, _ = evaluate(  # 评估
                model, test_loader, criterion, device
            )
            
            scheduler.step()  # 更新学习率
            
            if test_acc > best_fold_acc:  # 如果当前准确率超过最佳
                best_fold_acc = test_acc  # 更新最佳准确率
                best_fold_auc = test_auc  # 更新最佳AUC
            
            if (epoch + 1) % 10 == 0:  # 每10个epoch打印一次
                print(f"Epoch [{epoch+1}/{epochs}] - "  # 打印训练信息
                      f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, AUC: {test_auc:.4f}")
        
        fold_results.append({  # 记录该折结果
            'accuracy': best_fold_acc,
            'auc': best_fold_auc,
            'f1': test_f1
        })
        
        print(f"Fold {fold + 1} 最佳准确率: {best_fold_acc:.4f}, AUC: {best_fold_auc:.4f}")  # 打印该折最佳结果
    
    avg_acc = np.mean([r['accuracy'] for r in fold_results])  # 计算平均准确率
    std_acc = np.std([r['accuracy'] for r in fold_results])  # 计算准确率标准差
    avg_auc = np.mean([r['auc'] for r in fold_results])  # 计算平均AUC
    avg_f1 = np.mean([r['f1'] for r in fold_results])  # 计算平均F1
    
    print(f"\n{'='*50}")  # 打印分隔线
    print("交叉验证结果:")  # 打印交叉验证结果标题
    print(f"平均准确率: {avg_acc:.4f} ± {std_acc:.4f}")  # 打印平均准确率
    print(f"平均AUC: {avg_auc:.4f}")  # 打印平均AUC
    print(f"平均F1: {avg_f1:.4f}")  # 打印平均F1
    print(f"{'='*50}")  # 打印分隔线
    
    cv_results = {  # 创建交叉验证结果字典
        'avg_accuracy': avg_acc,
        'std_accuracy': std_acc,
        'avg_auc': avg_auc,
        'avg_f1': avg_f1,
        'fold_results': fold_results
    }
    
    with open('results/cv_results.json', 'w') as f:  # 保存交叉验证结果到JSON文件
        json.dump(cv_results, f, indent=2)
    
    return cv_results  # 返回交叉验证结果


def plot_training_history(history, save_dir='results'):  # 定义绘制训练历史曲线的函数
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 创建2x2的子图
    
    axes[0, 0].plot(history['train_loss'], label='Train Loss')  # 绘制训练损失曲线
    axes[0, 0].plot(history['test_loss'], label='Test Loss')  # 绘制测试损失曲线
    axes[0, 0].set_xlabel('Epoch')  # 设置x轴标签
    axes[0, 0].set_ylabel('Loss')  # 设置y轴标签
    axes[0, 0].set_title('Training and Test Loss')  # 设置标题
    axes[0, 0].legend()  # 显示图例
    axes[0, 0].grid(True)  # 显示网格
    
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy')  # 绘制训练准确率曲线
    axes[0, 1].plot(history['test_acc'], label='Test Accuracy')  # 绘制测试准确率曲线
    axes[0, 1].set_xlabel('Epoch')  # 设置x轴标签
    axes[0, 1].set_ylabel('Accuracy')  # 设置y轴标签
    axes[0, 1].set_title('Training and Test Accuracy')  # 设置标题
    axes[0, 1].legend()  # 显示图例
    axes[0, 1].grid(True)  # 显示网格
    
    axes[1, 0].plot(history['test_auc'], label='Test AUC', color='green')  # 绘制测试AUC曲线
    axes[1, 0].set_xlabel('Epoch')  # 设置x轴标签
    axes[1, 0].set_ylabel('AUC')  # 设置y轴标签
    axes[1, 0].set_title('Test AUC')  # 设置标题
    axes[1, 0].legend()  # 显示图例
    axes[1, 0].grid(True)  # 显示网格
    
    axes[1, 1].plot(history['test_f1'], label='Test F1', color='orange')  # 绘制测试F1曲线
    axes[1, 1].set_xlabel('Epoch')  # 设置x轴标签
    axes[1, 1].set_ylabel('F1 Score')  # 设置y轴标签
    axes[1, 1].set_title('Test F1 Score')  # 设置标题
    axes[1, 1].legend()  # 显示图例
    axes[1, 1].grid(True)  # 显示网格
    
    plt.tight_layout()  # 调整子图间距
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')  # 保存图像
    plt.close()  # 关闭图像


if __name__ == '__main__':  # 主程序入口
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置设备，优先使用GPU
    print(f"使用设备: {device}")  # 打印使用的设备
    
    data_path = 'data_3000.csv'  # 数据文件路径
    mirna_seq_path = 'miRNA+seq.csv'  # miRNA序列文件路径
    drug_smiles_path = 'drug+smiles.csv'  # 药物SMILES文件路径
    
    print("="*60)  # 打印分隔线
    print("开始训练模型...")  # 打印开始训练信息
    print("="*60)  # 打印分隔线
    
    model, history = train_model(  # 调用训练函数
        data_path=data_path,
        mirna_seq_path=mirna_seq_path,
        drug_smiles_path=drug_smiles_path,
        epochs=200,
        batch_size=32,
        lr=0.0005,
        negative_ratio=1,
        device=device,
        save_dir='results'
    )
    
    plot_training_history(history, save_dir='results')  # 绘制训练历史曲线
    
    print("\n" + "="*60)  # 打印分隔线
    print("进行5折交叉验证...")  # 打印开始交叉验证信息
    print("="*60)  # 打印分隔线
    
    cv_results = cross_validation(  # 调用交叉验证函数
        data_path=data_path,
        mirna_seq_path=mirna_seq_path,
        drug_smiles_path=drug_smiles_path,
        n_splits=5,
        epochs=150,
        batch_size=32,
        lr=0.0005,
        negative_ratio=1,
        device=device
    )
    
    print("\n所有训练完成!")  # 打印训练完成信息
