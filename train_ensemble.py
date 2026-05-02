import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.optim as optim  # 导入PyTorch优化器模块
from torch.utils.data import Dataset, DataLoader  # 导入数据集和数据加载器
import numpy as np  # 导入NumPy数值计算库
import pandas as pd  # 导入Pandas数据分析库
import os  # 导入操作系统模块
import json  # 导入JSON模块
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score  # 导入评估指标
from sklearn.model_selection import StratifiedKFold  # 导入分层K折交叉验证
import warnings  # 导入警告模块
warnings.filterwarnings('ignore')  # 过滤并忽略所有警告信息


def set_seed(seed=42):  # 定义设置随机种子的函数
    np.random.seed(seed)  # 设置NumPy随机种子
    torch.manual_seed(seed)  # 设置PyTorch CPU随机种子
    if torch.cuda.is_available():  # 如果CUDA可用
        torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子


class TransformerEncoder(nn.Module):  # 定义Transformer编码器类
    def __init__(self, vocab_size=5, d_model=128, nhead=8, num_layers=3,  # 初始化方法
                 dim_feedforward=256, dropout=0.1, output_dim=128, max_len=100):
        super(TransformerEncoder, self).__init__()  # 调用父类初始化方法
        
        self.embedding = nn.Embedding(vocab_size, d_model)  # 创建词嵌入层
        
        pe = torch.zeros(max_len, d_model)  # 创建位置编码矩阵
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 生成位置索引
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  # 计算分母项
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度使用sin函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度使用cos函数
        pe = pe.unsqueeze(0)  # 增加batch维度
        self.register_buffer('pe', pe)  # 注册为缓冲区
        
        encoder_layer = nn.TransformerEncoderLayer(  # 创建Transformer编码器层
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # 创建编码器
        
        self.fc = nn.Sequential(  # 创建全连接层
            nn.Linear(d_model, d_model),  # 线性变换
            nn.LayerNorm(d_model),  # 层归一化
            nn.ReLU(),  # ReLU激活函数
            nn.Dropout(dropout),  # Dropout层
            nn.Linear(d_model, output_dim)  # 输出层
        )
        
    def forward(self, x):  # 前向传播方法
        x = self.embedding(x)  # 词嵌入
        x = x + self.pe[:, :x.size(1), :]  # 添加位置编码
        x = self.transformer_encoder(x)  # 通过Transformer编码器
        x = x.mean(dim=1)  # 对序列维度求平均
        return self.fc(x)  # 通过全连接层输出


class CNNEncoder(nn.Module):  # 定义CNN编码器类
    def __init__(self, vocab_size=5, embed_dim=128, num_filters=64,  # 初始化方法
                 filter_sizes=[3, 5, 7], output_dim=128, dropout=0.1):
        super(CNNEncoder, self).__init__()  # 调用父类初始化方法
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # 创建词嵌入层
        
        self.convs = nn.ModuleList([  # 创建卷积层列表
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k//2)  # 不同卷积核大小的卷积层
            for k in filter_sizes
        ])
        
        self.fc = nn.Sequential(  # 创建全连接层
            nn.Linear(num_filters * len(filter_sizes), output_dim),  # 线性变换
            nn.ReLU(),  # ReLU激活函数
            nn.Dropout(dropout)  # Dropout层
        )
        
    def forward(self, x):  # 前向传播方法
        x = self.embedding(x)  # 词嵌入
        x = x.transpose(1, 2)  # 转置维度，适应Conv1d输入格式
        x = [torch.relu(conv(x)).max(dim=2)[0] for conv in self.convs]  # 对每个卷积层进行卷积和最大池化
        x = torch.cat(x, dim=1)  # 拼接所有卷积结果
        return self.fc(x)  # 通过全连接层输出


class LSTMEncoder(nn.Module):  # 定义LSTM编码器类
    def __init__(self, vocab_size=5, embed_dim=128, hidden_dim=128,  # 初始化方法
                 num_layers=2, output_dim=128, dropout=0.1, bidirectional=True):
        super(LSTMEncoder, self).__init__()  # 调用父类初始化方法
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # 创建词嵌入层
        
        self.lstm = nn.LSTM(  # 创建LSTM层
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim  # 计算LSTM输出维度
        self.fc = nn.Sequential(  # 创建全连接层
            nn.Linear(lstm_output_dim, output_dim),  # 线性变换
            nn.ReLU(),  # ReLU激活函数
            nn.Dropout(dropout)  # Dropout层
        )
        
    def forward(self, x):  # 前向传播方法
        x = self.embedding(x)  # 词嵌入
        lstm_out, (hidden, cell) = self.lstm(x)  # 通过LSTM层
        x = lstm_out[:, -1, :]  # 取最后一个时间步的输出
        return self.fc(x)  # 通过全连接层输出


class EnsembleModel(nn.Module):  # 定义集成模型类
    def __init__(self, drug_feat_dim=64, hidden_dim=256, dropout=0.2):  # 初始化方法
        super(EnsembleModel, self).__init__()  # 调用父类初始化方法
        
        self.transformer_encoder = TransformerEncoder(  # 创建Transformer编码器
            vocab_size=5, d_model=128, nhead=8, num_layers=3,
            dim_feedforward=256, dropout=dropout, output_dim=128
        )
        
        self.cnn_encoder = CNNEncoder(  # 创建CNN编码器
            vocab_size=5, embed_dim=128, num_filters=64,
            filter_sizes=[3, 5, 7], output_dim=128, dropout=dropout
        )
        
        self.lstm_encoder = LSTMEncoder(  # 创建LSTM编码器
            vocab_size=5, embed_dim=128, hidden_dim=128,
            num_layers=2, output_dim=128, dropout=dropout, bidirectional=True
        )
        
        self.drug_fc = nn.Sequential(  # 创建药物特征处理层
            nn.Linear(drug_feat_dim, 128),  # 线性变换
            nn.LayerNorm(128),  # 层归一化
            nn.ReLU(),  # ReLU激活函数
            nn.Dropout(dropout),  # Dropout层
            nn.Linear(128, 128)  # 输出层
        )
        
        self.attention = nn.Sequential(  # 创建注意力层，用于融合不同编码器的输出
            nn.Linear(128 * 3, 128),  # 线性变换
            nn.Tanh(),  # Tanh激活函数
            nn.Linear(128, 3),  # 输出注意力权重
            nn.Softmax(dim=1)  # Softmax归一化
        )
        
        self.bilinear = nn.Bilinear(128, 128, 128)  # 双线性融合层
        
        self.fusion = nn.Sequential(  # 创建融合层
            nn.Linear(128 * 2, hidden_dim),  # 线性变换
            nn.LayerNorm(hidden_dim),  # 层归一化
            nn.ReLU(),  # ReLU激活函数
            nn.Dropout(dropout),  # Dropout层
            nn.Linear(hidden_dim, 128),  # 线性变换
            nn.LayerNorm(128),  # 层归一化
            nn.ReLU(),  # ReLU激活函数
            nn.Dropout(dropout)  # Dropout层
        )
        
        self.classifier = nn.Sequential(  # 创建分类器
            nn.Linear(128, 64),  # 线性变换
            nn.ReLU(),  # ReLU激活函数
            nn.Dropout(dropout),  # Dropout层
            nn.Linear(64, 1)  # 输出层
        )
        
    def forward(self, mirna_seq, drug_feat):  # 前向传播方法
        transformer_out = self.transformer_encoder(mirna_seq)  # Transformer编码
        cnn_out = self.cnn_encoder(mirna_seq)  # CNN编码
        lstm_out = self.lstm_encoder(mirna_seq)  # LSTM编码
        
        stacked = torch.stack([transformer_out, cnn_out, lstm_out], dim=1)  # 堆叠三种编码器输出
        
        combined = torch.cat([transformer_out, cnn_out, lstm_out], dim=-1)  # 拼接三种编码器输出
        attn_weights = self.attention(combined)  # 计算注意力权重
        
        mirna_feat = torch.sum(stacked * attn_weights.unsqueeze(-1), dim=1)  # 加权求和得到miRNA特征
        
        drug_feat = self.drug_fc(drug_feat)  # 处理药物特征
        
        bilinear_out = self.bilinear(mirna_feat, drug_feat)  # 双线性融合
        
        fused = self.fusion(torch.cat([mirna_feat, bilinear_out], dim=-1))  # 融合特征
        return self.classifier(fused).squeeze(-1)  # 分类输出


class PrecomputedDataset(Dataset):  # 定义预计算数据集类
    def __init__(self, data_path, mirna_seq_path, drug_feat_path,  # 初始化方法
                 negative_ratio=1, max_seq_len=100, is_train=True, train_ratio=0.8):
        
        positive_data = pd.read_csv(data_path, header=None)  # 读取正样本数据
        positive_data.columns = ['miRNA', 'drug', 'pubchem_id']  # 设置列名
        positive_data['label'] = 1  # 设置标签为1
        
        mirna_seq_df = pd.read_csv(mirna_seq_path, header=None)  # 读取miRNA序列数据
        mirna_seq_df.columns = ['miRNA', 'sequence']  # 设置列名
        self.mirna_seq_dict = dict(zip(mirna_seq_df['miRNA'], mirna_seq_df['sequence']))  # 创建字典
        
        drug_feat_df = pd.read_csv(drug_feat_path, header=None)  # 读取药物特征数据
        self.drug_feat = drug_feat_df.values  # 保存为NumPy数组
        
        all_mirnas = positive_data['miRNA'].unique()  # 获取所有miRNA
        all_drugs = positive_data['drug'].unique()  # 获取所有药物
        positive_pairs = set(zip(positive_data['miRNA'], positive_data['drug']))  # 创建正样本对集合
        
        self.mirna_to_idx = {m: i for i, m in enumerate(all_mirnas)}  # 创建miRNA索引映射
        self.drug_to_idx = {d: i for i, d in enumerate(all_drugs)}  # 创建药物索引映射
        
        np.random.seed(42)  # 设置随机种子
        negative_samples = []  # 初始化负样本列表
        for _ in range(len(positive_data) * negative_ratio):  # 循环生成负样本
            while True:  # 持续循环直到生成有效负样本
                mirna = np.random.choice(all_mirnas)  # 随机选择miRNA
                drug = np.random.choice(all_drugs)  # 随机选择药物
                if (mirna, drug) not in positive_pairs:  # 如果不是正样本对
                    negative_samples.append({  # 添加到负样本列表
                        'miRNA': mirna, 'drug': drug, 'label': 0,
                        'mirna_idx': self.mirna_to_idx[mirna],
                        'drug_idx': self.drug_to_idx.get(drug, 0)
                    })
                    break  # 跳出循环
        
        negative_data = pd.DataFrame(negative_samples)  # 创建负样本数据框
        
        positive_data['mirna_idx'] = positive_data['miRNA'].map(self.mirna_to_idx)  # 添加miRNA索引
        positive_data['drug_idx'] = positive_data['drug'].map(lambda x: self.drug_to_idx.get(x, 0))  # 添加药物索引
        
        self.data = pd.concat([positive_data, negative_data], ignore_index=True)  # 合并正负样本
        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)  # 随机打乱
        
        n_total = len(self.data)  # 获取总样本数
        n_train = int(n_total * train_ratio)  # 计算训练集大小
        
        if is_train:  # 如果是训练集
            self.data = self.data[:n_train]  # 取前n_train个样本
        else:  # 如果是测试集
            self.data = self.data[n_train:]  # 取剩余样本
        
        self.max_seq_len = max_seq_len  # 保存最大序列长度
        self.nuc_to_idx = {'A': 1, 'U': 2, 'G': 3, 'C': 4, 'N': 0}  # 核苷酸索引映射
        
        print(f"数据集大小: {len(self.data)}")  # 打印数据集大小
        
    def _encode_mirna(self, seq):  # 定义miRNA编码方法
        seq = seq.upper().replace('T', 'U')  # 转换为大写并将T替换为U
        encoded = [self.nuc_to_idx.get(n, 0) for n in seq]  # 将核苷酸转换为索引
        
        if len(encoded) < self.max_seq_len:  # 如果序列长度小于最大长度
            encoded = encoded + [0] * (self.max_seq_len - len(encoded))  # 填充0
        else:  # 如果序列长度大于等于最大长度
            encoded = encoded[:self.max_seq_len]  # 截断
        
        return torch.LongTensor(encoded)  # 返回LongTensor
    
    def __len__(self):  # 定义获取数据集长度的方法
        return len(self.data)  # 返回数据集大小
    
    def __getitem__(self, idx):  # 定义获取单个样本的方法
        row = self.data.iloc[idx]  # 获取指定索引的行
        
        seq = self.mirna_seq_dict.get(row['miRNA'], '')  # 获取miRNA序列
        mirna_encoded = self._encode_mirna(seq)  # 编码序列
        
        drug_feat = torch.FloatTensor(self.drug_feat[row['drug_idx']])  # 获取药物特征
        
        return {  # 返回样本字典
            'mirna': mirna_encoded,
            'drug_feat': drug_feat,
            'label': torch.FloatTensor([row['label']])
        }


def train_epoch(model, dataloader, optimizer, criterion, device):  # 定义训练一个epoch的函数
    model.train()  # 设置为训练模式
    total_loss = 0  # 初始化总损失
    all_preds, all_labels = [], []  # 初始化预测和标签列表
    
    for batch in dataloader:  # 遍历数据加载器
        mirna = batch['mirna'].to(device)  # 将miRNA数据移到设备
        drug_feat = batch['drug_feat'].to(device)  # 将药物特征移到设备
        labels = batch['label'].squeeze(-1).to(device)  # 将标签移到设备
        
        optimizer.zero_grad()  # 清零梯度
        outputs = model(mirna, drug_feat)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()  # 更新参数
        
        total_loss += loss.item()  # 累加损失
        preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()  # 计算预测
        all_preds.extend(preds)  # 添加预测结果
        all_labels.extend(labels.cpu().numpy())  # 添加标签
    
    return total_loss / len(dataloader), accuracy_score(all_labels, all_preds)  # 返回平均损失和准确率


def evaluate(model, dataloader, criterion, device):  # 定义评估函数
    model.eval()  # 设置为评估模式
    total_loss = 0  # 初始化总损失
    all_preds, all_labels, all_probs = [], [], []  # 初始化列表
    
    with torch.no_grad():  # 禁用梯度计算
        for batch in dataloader:  # 遍历数据加载器
            mirna = batch['mirna'].to(device)  # 将miRNA数据移到设备
            drug_feat = batch['drug_feat'].to(device)  # 将药物特征移到设备
            labels = batch['label'].squeeze(-1).to(device)  # 将标签移到设备
            
            outputs = model(mirna, drug_feat)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            
            total_loss += loss.item()  # 累加损失
            probs = torch.sigmoid(outputs).cpu().numpy()  # 计算概率
            all_probs.extend(probs)  # 添加概率
            all_preds.extend((probs > 0.5).astype(float))  # 添加预测
            all_labels.extend(labels.cpu().numpy())  # 添加标签
    
    return (  # 返回评估结果
        total_loss / len(dataloader),
        accuracy_score(all_labels, all_preds),
        roc_auc_score(all_labels, all_probs),
        f1_score(all_labels, all_preds),
        precision_score(all_labels, all_preds, zero_division=0),
        recall_score(all_labels, all_preds, zero_division=0)
    )


def train():  # 定义训练函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置设备
    print(f"使用设备: {device}")  # 打印设备信息
    
    train_dataset = PrecomputedDataset(  # 创建训练数据集
        'data_3000.csv', 'miRNA+seq.csv', 'drug_GIN_64.csv',
        negative_ratio=1, is_train=True
    )
    test_dataset = PrecomputedDataset(  # 创建测试数据集
        'data_3000.csv', 'miRNA+seq.csv', 'drug_GIN_64.csv',
        negative_ratio=1, is_train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 创建训练数据加载器
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 创建测试数据加载器
    
    model = EnsembleModel(  # 创建集成模型
        drug_feat_dim=64, hidden_dim=256, dropout=0.2
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()  # 定义损失函数
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)  # 定义优化器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)  # 定义学习率调度器
    
    best_acc = 0  # 初始化最佳准确率
    best_auc = 0  # 初始化最佳AUC
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'test_auc': []}  # 初始化历史记录
    
    print("\n开始训练集成模型...")  # 打印开始训练信息
    for epoch in range(200):  # 遍历每个epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)  # 训练
        test_loss, test_acc, test_auc, test_f1, test_precision, test_recall = evaluate(  # 评估
            model, test_loader, criterion, device
        )
        scheduler.step()  # 更新学习率
        
        history['train_loss'].append(train_loss)  # 记录训练损失
        history['train_acc'].append(train_acc)  # 记录训练准确率
        history['test_loss'].append(test_loss)  # 记录测试损失
        history['test_acc'].append(test_acc)  # 记录测试准确率
        history['test_auc'].append(test_auc)  # 记录测试AUC
        
        if test_acc > best_acc:  # 如果当前准确率更好
            best_acc = test_acc  # 更新最佳准确率
            best_auc = test_auc  # 更新最佳AUC
            torch.save(model.state_dict(), 'results/best_ensemble_model.pth')  # 保存模型
        
        if (epoch + 1) % 10 == 0:  # 每10个epoch打印一次
            print(f"Epoch [{epoch+1}/200] Train: {train_acc:.4f}, Test: {test_acc:.4f}, AUC: {test_auc:.4f}, F1: {test_f1:.4f}")
    
    print(f"\n最佳测试准确率: {best_acc:.4f}")  # 打印最佳准确率
    print(f"最佳测试AUC: {best_auc:.4f}")  # 打印最佳AUC
    
    with open('results/ensemble_history.json', 'w') as f:  # 保存训练历史
        json.dump(history, f, indent=2)
    
    return model, history, best_acc  # 返回模型、历史和最佳准确率


def cross_validation(n_splits=5, epochs=150):  # 定义交叉验证函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置设备
    print(f"使用设备: {device}")  # 打印设备信息
    
    dataset = PrecomputedDataset(  # 创建完整数据集
        'data_3000.csv', 'miRNA+seq.csv', 'drug_GIN_64.csv',
        negative_ratio=1, is_train=True, train_ratio=1.0
    )
    
    labels = dataset.data['label'].values  # 获取标签
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)  # 创建分层K折交叉验证器
    
    fold_results = []  # 初始化折结果列表
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):  # 遍历每一折
        print(f"\n{'='*50}")  # 打印分隔线
        print(f"Fold {fold + 1}/{n_splits}")  # 打印当前折数
        print(f"{'='*50}")  # 打印分隔线
        
        train_subset = torch.utils.data.Subset(dataset, train_idx)  # 创建训练子集
        test_subset = torch.utils.data.Subset(dataset, test_idx)  # 创建测试子集
        
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)  # 创建训练数据加载器
        test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)  # 创建测试数据加载器
        
        model = EnsembleModel(  # 创建集成模型
            drug_feat_dim=64, hidden_dim=256, dropout=0.2
        ).to(device)
        
        criterion = nn.BCEWithLogitsLoss()  # 定义损失函数
        optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)  # 定义优化器
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)  # 定义学习率调度器
        
        best_fold_acc = 0  # 初始化该折最佳准确率
        best_fold_auc = 0  # 初始化该折最佳AUC
        
        for epoch in range(epochs):  # 遍历每个epoch
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)  # 训练
            test_loss, test_acc, test_auc, test_f1, _, _ = evaluate(  # 评估
                model, test_loader, criterion, device
            )
            scheduler.step()  # 更新学习率
            
            if test_acc > best_fold_acc:  # 如果当前准确率更好
                best_fold_acc = test_acc  # 更新最佳准确率
                best_fold_auc = test_auc  # 更新最佳AUC
            
            if (epoch + 1) % 15 == 0:  # 每15个epoch打印一次
                print(f"Epoch [{epoch+1}/{epochs}] Train: {train_acc:.4f}, Test: {test_acc:.4f}, AUC: {test_auc:.4f}")
        
        fold_results.append({'accuracy': best_fold_acc, 'auc': best_fold_auc, 'f1': test_f1})  # 记录该折结果
        print(f"Fold {fold + 1} 最佳: Acc={best_fold_acc:.4f}, AUC={best_fold_auc:.4f}")  # 打印该折最佳结果
    
    avg_acc = np.mean([r['accuracy'] for r in fold_results])  # 计算平均准确率
    std_acc = np.std([r['accuracy'] for r in fold_results])  # 计算准确率标准差
    avg_auc = np.mean([r['auc'] for r in fold_results])  # 计算平均AUC
    avg_f1 = np.mean([r['f1'] for r in fold_results])  # 计算平均F1
    
    print(f"\n{'='*50}")  # 打印分隔线
    print(f"交叉验证结果: {avg_acc:.4f} ± {std_acc:.4f}")  # 打印交叉验证结果
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
    
    with open('results/ensemble_cv_results.json', 'w') as f:  # 保存交叉验证结果
        json.dump(cv_results, f, indent=2)
    
    return cv_results  # 返回交叉验证结果


if __name__ == '__main__':  # 主程序入口
    os.makedirs('results', exist_ok=True)  # 创建结果目录
    set_seed(42)  # 设置随机种子
    
    print("="*60)  # 打印分隔线
    print("训练集成模型...")  # 打印训练信息
    print("="*60)  # 打印分隔线
    model, history, best_acc = train()  # 调用训练函数
    
    print("\n" + "="*60)  # 打印分隔线
    print("进行5折交叉验证...")  # 打印交叉验证信息
    print("="*60)  # 打印分隔线
    cv_results = cross_validation(n_splits=5, epochs=150)  # 调用交叉验证函数
    
    print("\n训练完成!")  # 打印训练完成信息
