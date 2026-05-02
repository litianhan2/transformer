import torch  # 导入PyTorch深度学习框架，用于构建和训练神经网络
import torch.nn as nn  # 导入PyTorch神经网络模块，包含各种层和损失函数
import torch.nn.functional as F  # 导入PyTorch函数式接口，包含激活函数等
import math  # 导入Python数学库，用于数学计算


class PositionalEncoding(nn.Module):  # 定义位置编码类，继承自nn.Module，用于为序列添加位置信息
    def __init__(self, d_model, max_len=500):  # 初始化方法，d_model为模型维度，max_len为最大序列长度
        super(PositionalEncoding, self).__init__()  # 调用父类初始化方法
        pe = torch.zeros(max_len, d_model)  # 创建位置编码矩阵，形状为(max_len, d_model)，初始化为0
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 生成位置索引，形状为(max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 计算位置编码的分母项，使用指数衰减
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度使用sin函数进行位置编码
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度使用cos函数进行位置编码
        pe = pe.unsqueeze(0)  # 在第0维增加一个维度，形状变为(1, max_len, d_model)
        self.register_buffer('pe', pe)  # 将位置编码注册为缓冲区，不参与梯度更新但会保存到模型状态中

    def forward(self, x):  # 前向传播方法，x为输入张量
        return x + self.pe[:, :x.size(1), :]  # 将位置编码加到输入上，只使用与输入序列长度相等的部分


class miRNATransformer(nn.Module):  # 定义miRNA序列编码器类，使用Transformer架构
    def __init__(self, vocab_size=5, d_model=128, nhead=8, num_layers=4,  # 初始化方法，vocab_size为词表大小(4种核苷酸+padding)
                 dim_feedforward=256, dropout=0.1, output_dim=128):  # d_model为模型维度，nhead为注意力头数，num_layers为编码器层数
        super(miRNATransformer, self).__init__()  # 调用父类初始化方法
        
        self.embedding = nn.Embedding(vocab_size, d_model)  # 创建词嵌入层，将核苷酸索引映射为d_model维向量
        self.pos_encoder = PositionalEncoding(d_model)  # 创建位置编码器，为序列添加位置信息
        
        encoder_layer = nn.TransformerEncoderLayer(  # 创建单个Transformer编码器层
            d_model=d_model,  # 模型维度
            nhead=nhead,  # 多头注意力的头数
            dim_feedforward=dim_feedforward,  # 前馈网络的隐藏层维度
            dropout=dropout,  # dropout概率
            batch_first=True  # 输入数据的batch维度在第一位
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # 创建完整的Transformer编码器，堆叠num_layers个编码器层
        
        self.fc = nn.Sequential(  # 创建全连接层序列，用于将编码器输出映射到最终输出维度
            nn.Linear(d_model, d_model),  # 第一层线性变换，维度不变
            nn.ReLU(),  # ReLU激活函数
            nn.Dropout(dropout),  # Dropout层，防止过拟合
            nn.Linear(d_model, output_dim)  # 第二层线性变换，输出维度为output_dim
        )
        
    def forward(self, x):  # 前向传播方法，x为输入的miRNA序列索引
        x = self.embedding(x)  # 将输入索引转换为词嵌入向量
        x = self.pos_encoder(x)  # 添加位置编码
        x = self.transformer_encoder(x)  # 通过Transformer编码器进行编码
        x = x.mean(dim=1)  # 对序列维度求平均，得到序列级别的表示
        x = self.fc(x)  # 通过全连接层得到最终输出
        return x  # 返回编码后的miRNA特征向量


class GraphConvLayer(nn.Module):  # 定义图卷积层类，用于处理图结构数据
    def __init__(self, in_features, out_features, dropout=0.1):  # 初始化方法，in_features为输入特征维度，out_features为输出特征维度
        super(GraphConvLayer, self).__init__()  # 调用父类初始化方法
        self.linear = nn.Linear(in_features, out_features)  # 创建线性变换层
        self.dropout = nn.Dropout(dropout)  # 创建Dropout层
        
    def forward(self, x, adj):  # 前向传播方法，x为节点特征，adj为邻接矩阵
        x = self.linear(x)  # 对节点特征进行线性变换
        x = torch.matmul(adj, x)  # 通过邻接矩阵进行消息传递，聚合邻居节点信息
        x = self.dropout(x)  # 应用Dropout
        return x  # 返回更新后的节点特征


class DrugGNN(nn.Module):  # 定义药物图神经网络类，用于编码药物分子图
    def __init__(self, node_features=29, hidden_dim=128, output_dim=128,  # 初始化方法，node_features为节点特征维度
                 num_layers=3, dropout=0.1):  # hidden_dim为隐藏层维度，num_layers为GNN层数
        super(DrugGNN, self).__init__()  # 调用父类初始化方法
        
        self.num_layers = num_layers  # 保存GNN层数
        
        self.input_fc = nn.Linear(node_features, hidden_dim)  # 创建输入线性层，将节点特征映射到隐藏维度
        
        self.gcn_layers = nn.ModuleList()  # 创建ModuleList存储GCN层
        self.batch_norms = nn.ModuleList()  # 创建ModuleList存储批归一化层
        
        for i in range(num_layers):  # 循环创建num_layers个GCN层
            self.gcn_layers.append(GraphConvLayer(hidden_dim, hidden_dim, dropout))  # 添加图卷积层
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))  # 添加批归一化层
        
        self.fc = nn.Sequential(  # 创建全连接层序列，用于生成图级别表示
            nn.Linear(hidden_dim * 2, hidden_dim),  # 输入维度为hidden_dim*2(拼接mean和max池化结果)
            nn.ReLU(),  # ReLU激活函数
            nn.Dropout(dropout),  # Dropout层
            nn.Linear(hidden_dim, output_dim)  # 输出层，输出维度为output_dim
        )
        
    def forward(self, node_features, adj):  # 前向传播方法，node_features为节点特征矩阵，adj为邻接矩阵
        h = self.input_fc(node_features)  # 将节点特征映射到隐藏维度
        
        all_h = [h]  # 保存所有层的隐藏状态，用于后续的多层特征融合
        for i in range(self.num_layers):  # 循环通过每一层GCN
            h = self.gcn_layers[i](h, adj)  # 通过第i层图卷积
            h = self.batch_norms[i](h)  # 应用批归一化
            h = F.relu(h)  # 应用ReLU激活函数
            all_h.append(h)  # 保存当前层的隐藏状态
        
        h_mean = torch.stack(all_h, dim=0).mean(dim=0)  # 对所有层的隐藏状态求平均
        h_max = torch.stack(all_h, dim=0).max(dim=0)[0]  # 对所有层的隐藏状态求最大值
        h = torch.cat([h_mean, h_max], dim=-1)  # 拼接平均池化和最大池化结果
        
        graph_embedding = h.mean(dim=0, keepdim=True)  # 对节点维度求平均，得到图级别嵌入，keepdim=True保持维度
        
        output = self.fc(graph_embedding)  # 通过全连接层得到最终输出
        return output  # 返回药物分子的编码向量


class BilinearFusion(nn.Module):  # 定义双线性融合类，用于融合miRNA和药物特征
    def __init__(self, dim1=128, dim2=128, output_dim=128, dropout=0.1):  # 初始化方法，dim1和dim2为两个输入维度
        super(BilinearFusion, self).__init__()  # 调用父类初始化方法
        self.bilinear = nn.Bilinear(dim1, dim2, output_dim)  # 创建双线性层，学习两个输入的交互特征
        self.dropout = nn.Dropout(dropout)  # 创建Dropout层
        
    def forward(self, x1, x2):  # 前向传播方法，x1和x2为两个输入特征
        out = self.bilinear(x1, x2)  # 通过双线性层计算交互特征
        out = F.relu(out)  # 应用ReLU激活函数
        out = self.dropout(out)  # 应用Dropout
        return out  # 返回融合后的特征


class miRNADrugResistanceModel(nn.Module):  # 定义miRNA-药物耐药性预测模型类
    def __init__(self, mirna_vocab_size=5, mirna_d_model=128, mirna_nhead=8,  # 初始化方法，设置miRNA编码器参数
                 mirna_num_layers=4, mirna_dim_ff=256,
                 drug_node_features=29, drug_hidden_dim=128, drug_num_layers=3,  # 设置药物编码器参数
                 fusion_dim=256, dropout=0.2):  # 设置融合层参数
        super(miRNADrugResistanceModel, self).__init__()  # 调用父类初始化方法
        
        self.mirna_encoder = miRNATransformer(  # 创建miRNA编码器
            vocab_size=mirna_vocab_size,  # 词表大小
            d_model=mirna_d_model,  # 模型维度
            nhead=mirna_nhead,  # 注意力头数
            num_layers=mirna_num_layers,  # 编码器层数
            dim_feedforward=mirna_dim_ff,  # 前馈网络维度
            dropout=dropout,  # dropout概率
            output_dim=128  # 输出维度
        )
        
        self.drug_encoder = DrugGNN(  # 创建药物编码器
            node_features=drug_node_features,  # 节点特征维度
            hidden_dim=drug_hidden_dim,  # 隐藏层维度
            output_dim=128,  # 输出维度
            num_layers=drug_num_layers,  # GNN层数
            dropout=dropout  # dropout概率
        )
        
        self.bilinear_fusion = BilinearFusion(128, 128, 128, dropout)  # 创建双线性融合层，融合miRNA和药物特征
        
        self.fusion = nn.Sequential(  # 创建融合层序列，整合所有特征
            nn.Linear(128 * 3, fusion_dim),  # 输入维度为384(拼接miRNA、药物和双线性融合特征)
            nn.ReLU(),  # ReLU激活函数
            nn.Dropout(dropout),  # Dropout层
            nn.Linear(fusion_dim, fusion_dim // 2),  # 降维到fusion_dim的一半
            nn.ReLU(),  # ReLU激活函数
            nn.Dropout(dropout)  # Dropout层
        )
        
        self.classifier = nn.Sequential(  # 创建分类器序列，用于最终预测
            nn.Linear(fusion_dim // 2, 64),  # 线性层降维到64
            nn.ReLU(),  # ReLU激活函数
            nn.Dropout(dropout),  # Dropout层
            nn.Linear(64, 1)  # 输出层，输出单个预测值
        )
        
    def forward(self, mirna_seq, drug_node_features, drug_adj):  # 前向传播方法
        mirna_feat = self.mirna_encoder(mirna_seq)  # 编码miRNA序列
        drug_feat = self.drug_encoder(drug_node_features, drug_adj)  # 编码药物分子图
        
        bilinear_out = self.bilinear_fusion(mirna_feat, drug_feat)  # 计算miRNA和药物的双线性融合特征
        
        combined = torch.cat([mirna_feat, drug_feat, bilinear_out], dim=-1)  # 拼接三种特征
        
        fused = self.fusion(combined)  # 通过融合层
        output = self.classifier(fused)  # 通过分类器得到预测结果
        
        return output.squeeze(-1)  # 去除最后一维，返回预测logits
