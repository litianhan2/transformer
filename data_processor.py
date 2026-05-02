import torch  # 导入PyTorch深度学习框架，用于构建和训练神经网络模型
import numpy as np  # 导入NumPy数值计算库，用于数组操作和数学计算
import pandas as pd  # 导入Pandas数据分析库，用于数据读取和处理
from torch.utils.data import Dataset  # 从PyTorch导入Dataset基类，用于自定义数据集类
from rdkit import Chem  # 导入RDKit化学信息学库的Chem模块，用于分子结构处理
from rdkit.Chem import AllChem, Descriptors  # 从RDKit导入AllChem和Descriptors模块，用于分子特征计算和描述符生成
import warnings  # 导入Python警告模块，用于控制警告信息的显示
warnings.filterwarnings('ignore')  # 过滤并忽略所有警告信息，避免输出干扰


class miRNADrugDataset(Dataset):  # 定义miRNADrugDataset类，继承自PyTorch的Dataset类，用于处理miRNA-药物关联数据
    def __init__(self, data_path, mirna_seq_path, drug_smiles_path,  # 初始化方法，接收数据文件路径和配置参数
                 mirna_kmer_path=None, drug_feat_path=None, max_seq_len=100):  # 可选参数：k-mer特征路径、药物特征路径、最大序列长度默认100
        self.data = pd.read_csv(data_path, header=None)  # 从CSV文件读取miRNA-药物关联数据，无表头
        self.data.columns = ['miRNA', 'drug', 'pubchem_id']  # 为数据框设置列名：miRNA名称、药物名称、PubChem ID
        self.data['label'] = 1  # 为所有样本添加标签列，值为1表示正样本（存在关联）
        
        self.mirna_seq = pd.read_csv(mirna_seq_path, header=None)  # 从CSV文件读取miRNA序列数据，无表头
        self.mirna_seq.columns = ['miRNA', 'sequence']  # 为miRNA序列数据框设置列名：miRNA名称、序列字符串
        self.mirna_seq_dict = dict(zip(self.mirna_seq['miRNA'], self.mirna_seq['sequence']))  # 创建miRNA名称到序列的字典映射，便于快速查找
        
        self.drug_smiles = pd.read_csv(drug_smiles_path, header=None)  # 从CSV文件读取药物SMILES字符串数据，无表头
        self.drug_smiles.columns = ['drug', 'smiles']  # 为药物SMILES数据框设置列名：药物名称、SMILES字符串
        self.drug_smiles_dict = dict(zip(self.drug_smiles['drug'], self.drug_smiles['smiles']))  # 创建药物名称到SMILES的字典映射，便于快速查找
        
        self.max_seq_len = max_seq_len  # 保存最大序列长度参数，用于序列填充或截断
        self.nucleotide_to_idx = {'A': 1, 'U': 2, 'G': 3, 'C': 4, 'N': 0}  # 定义核苷酸到索引的映射字典，A=1,U=2,G=3,C=4,N/未知=0
        
        self.drug_graphs = {}  # 初始化药物图字典，用于存储药物的分子图表示（节点特征和邻接矩阵）
        self._build_drug_graphs()  # 调用内部方法构建所有药物的分子图表示
        
        self.mirna_cache = {}  # 初始化miRNA序列编码缓存字典，避免重复编码相同序列
        
    def _build_drug_graphs(self):  # 定义内部方法，用于构建所有药物的分子图表示
        for drug, smiles in self.drug_smiles_dict.items():  # 遍历所有药物的名称和对应的SMILES字符串
            mol = Chem.MolFromSmiles(smiles)  # 使用RDKit从SMILES字符串解析分子对象
            if mol is None:  # 如果分子对象解析失败（SMILES格式错误）
                continue  # 跳过当前药物，继续处理下一个
            
            num_atoms = mol.GetNumAtoms()  # 获取分子中的原子数量
            adj = Chem.GetAdjacencyMatrix(mol)  # 获取分子的邻接矩阵，表示原子之间的连接关系
            adj = adj + np.eye(num_atoms)  # 在邻接矩阵上加上单位矩阵，实现自环，即每个原子与自身相连
            adj = torch.FloatTensor(adj)  # 将邻接矩阵转换为PyTorch浮点张量，用于神经网络计算
            
            node_features = []  # 初始化节点特征列表，用于存储所有原子的特征向量
            for atom in mol.GetAtoms():  # 遍历分子中的所有原子
                features = self._get_atom_features(atom)  # 调用方法获取当前原子的特征向量
                node_features.append(features)  # 将原子特征添加到列表中
            
            if len(node_features) > 0:  # 如果成功提取了原子特征（列表不为空）
                node_features = torch.FloatTensor(node_features)  # 将节点特征列表转换为PyTorch浮点张量
                self.drug_graphs[drug] = (node_features, adj)  # 将药物名称映射到(节点特征, 邻接矩阵)元组
    
    def _get_atom_features(self, atom):  # 定义内部方法，用于提取单个原子的特征向量
        features = []  # 初始化特征列表，用于存储原子的所有特征
        
        atom_type = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'other']  # 定义原子类型列表，包含常见原子和一个"其他"类别
        atom_symbol = atom.GetSymbol()  # 获取原子的化学符号（如'C', 'N', 'O'等）
        atom_feat = [0] * len(atom_type)  # 初始化原子类型特征向量，长度与原子类型列表相同，初始值全为0
        if atom_symbol in atom_type:  # 如果原子符号在预定义的原子类型列表中
            atom_feat[atom_type.index(atom_symbol)] = 1  # 将对应位置设为1，实现one-hot编码
        else:  # 如果原子符号不在预定义列表中
            atom_feat[-1] = 1  # 将最后一个位置（"other"类别）设为1
        features.extend(atom_feat)  # 将原子类型特征添加到总特征列表中
        
        degree = [0, 1, 2, 3, 4, 5]  # 定义原子度数（连接的化学键数量）的可能值列表
        d = atom.GetDegree()  # 获取当前原子的度数
        degree_feat = [0] * len(degree)  # 初始化度数特征向量，长度与度数列表相同，初始值全为0
        if d < len(degree):  # 如果度数在预定义范围内
            degree_feat[d] = 1  # 将对应位置设为1，实现one-hot编码
        features.extend(degree_feat)  # 将度数特征添加到总特征列表中
        
        hybridization = [Chem.rdchem.HybridizationType.SP,  # 定义杂化类型列表，包含SP杂化
                        Chem.rdchem.HybridizationType.SP2,  # SP2杂化
                        Chem.rdchem.HybridizationType.SP3,  # SP3杂化
                        Chem.rdchem.HybridizationType.SP3D,  # SP3D杂化
                        Chem.rdchem.HybridizationType.SP3D2]  # SP3D2杂化
        h = atom.GetHybridization()  # 获取当前原子的杂化类型
        hybrid_feat = [0] * len(hybridization)  # 初始化杂化特征向量，长度与杂化类型列表相同，初始值全为0
        if h in hybridization:  # 如果杂化类型在预定义列表中
            hybrid_feat[hybridization.index(h)] = 1  # 将对应位置设为1，实现one-hot编码
        features.extend(hybrid_feat)  # 将杂化特征添加到总特征列表中
        
        features.append(atom.GetIsAromatic())  # 添加原子是否为芳香族原子的布尔值特征（True/False）
        features.append(atom.HasQuery())  # 添加原子是否有查询属性的布尔值特征（用于复杂分子定义）
        
        formal_charge = atom.GetFormalCharge()  # 获取原子的形式电荷
        features.append(formal_charge)  # 将形式电荷添加到特征列表中
        
        num_hs = atom.GetTotalNumHs()  # 获取原子连接的氢原子总数
        features.append(num_hs)  # 将氢原子数量添加到特征列表中
        
        features.append(atom.IsInRing())  # 添加原子是否在环结构中的布尔值特征
        
        features.append(atom.GetMass() / 100.0)  # 添加原子质量（归一化处理，除以100），便于神经网络学习
        
        features.append(atom.GetExplicitValence())  # 添加原子的显式化合价特征
        features.append(atom.GetImplicitValence())  # 添加原子的隐式化合价特征
        
        return features  # 返回包含所有特征的列表
    
    def _encode_mirna_sequence(self, sequence):  # 定义内部方法，用于对miRNA序列进行编码
        sequence = sequence.upper()  # 将序列转换为大写字母，统一格式
        sequence = sequence.replace('T', 'U')  # 将序列中的T（胸腺嘧啶）替换为U（尿嘧啶），因为miRNA是RNA序列
        
        encoded = [self.nucleotide_to_idx.get(nuc, 0) for nuc in sequence]  # 将序列中的每个核苷酸转换为对应的索引值，未知核苷酸默认为0
        
        if len(encoded) < self.max_seq_len:  # 如果编码后的序列长度小于最大序列长度
            encoded = encoded + [0] * (self.max_seq_len - len(encoded))  # 在序列末尾填充0，使长度达到max_seq_len
        else:  # 如果编码后的序列长度大于或等于最大序列长度
            encoded = encoded[:self.max_seq_len]  # 截断序列，只保留前max_seq_len个核苷酸
        
        return torch.LongTensor(encoded)  # 将编码后的序列转换为PyTorch长整型张量并返回
    
    def __len__(self):  # 定义特殊方法，返回数据集的样本总数
        return len(self.data)  # 返回数据框的行数，即样本数量
    
    def __getitem__(self, idx):  # 定义特殊方法，根据索引获取单个样本
        row = self.data.iloc[idx]  # 根据索引获取数据框中的一行数据
        mirna_name = row['miRNA']  # 获取该样本的miRNA名称
        drug_name = row['drug']  # 获取该样本的药物名称
        label = row['label']  # 获取该样本的标签（1表示正样本）
        
        if mirna_name in self.mirna_cache:  # 如果miRNA名称已在缓存中
            mirna_seq_encoded = self.mirna_cache[mirna_name]  # 直接从缓存获取编码后的序列
        else:  # 如果miRNA名称不在缓存中
            sequence = self.mirna_seq_dict.get(mirna_name, '')  # 从字典获取miRNA序列，如果不存在则返回空字符串
            mirna_seq_encoded = self._encode_mirna_sequence(sequence)  # 对序列进行编码
            self.mirna_cache[mirna_name] = mirna_seq_encoded  # 将编码结果存入缓存，避免重复计算
        
        if drug_name not in self.drug_graphs:  # 如果药物名称不在药物图字典中（SMILES解析失败的情况）
            drug_name = list(self.drug_graphs.keys())[0]  # 使用第一个可用的药物作为替代
        
        node_features, adj = self.drug_graphs[drug_name]  # 从药物图字典获取节点特征和邻接矩阵
        
        return {  # 返回包含样本所有信息的字典
            'mirna_seq': mirna_seq_encoded,  # 编码后的miRNA序列张量
            'drug_node_features': node_features,  # 药物分子的节点特征张量
            'drug_adj': adj,  # 药物分子的邻接矩阵张量
            'label': torch.FloatTensor([label]),  # 标签张量，转换为浮点型用于损失计算
            'mirna_name': mirna_name,  # miRNA名称字符串
            'drug_name': drug_name  # 药物名称字符串
        }


def collate_fn(batch):  # 定义批处理函数，用于DataLoader将多个样本整合成一个批次
    mirna_seqs = torch.stack([item['mirna_seq'] for item in batch])  # 将批次中所有miRNA序列堆叠成一个张量
    labels = torch.stack([item['label'] for item in batch])  # 将批次中所有标签堆叠成一个张量
    mirna_names = [item['mirna_name'] for item in batch]  # 提取批次中所有miRNA名称，保持为列表形式
    drug_names = [item['drug_name'] for item in batch]  # 提取批次中所有药物名称，保持为列表形式
    
    return {  # 返回包含批次数据的字典
        'mirna_seqs': mirna_seqs,  # 批次miRNA序列张量，形状为(batch_size, seq_len)
        'drug_data': [(item['drug_node_features'], item['drug_adj']) for item in batch],  # 药物数据列表，每个元素是(节点特征, 邻接矩阵)元组
        'labels': labels,  # 批次标签张量，形状为(batch_size, 1)
        'mirna_names': mirna_names,  # miRNA名称列表
        'drug_names': drug_names  # 药物名称列表
    }


def create_negative_samples(data_path, mirna_seq_path, drug_smiles_path,  # 定义创建负样本的函数，用于生成不存在的miRNA-药物对
                           negative_ratio=1, max_seq_len=100):  # 参数：负样本比例默认1:1，最大序列长度默认100
    data = pd.read_csv(data_path, header=None)  # 从CSV文件读取正样本数据，无表头
    data.columns = ['miRNA', 'drug', 'pubchem_id']  # 为数据框设置列名：miRNA名称、药物名称、PubChem ID
    
    all_mirnas = data['miRNA'].unique()  # 获取所有唯一的miRNA名称
    all_drugs = data['drug'].unique()  # 获取所有唯一的药物名称
    
    positive_pairs = set(zip(data['miRNA'], data['drug']))  # 创建正样本对的集合，用于检查是否已存在关联
    
    negative_samples = []  # 初始化负样本列表
    for _ in range(len(data) * negative_ratio):  # 循环生成指定数量的负样本（正样本数量 × 负样本比例）
        while True:  # 持续循环直到生成有效的负样本
            mirna = np.random.choice(all_mirnas)  # 随机选择一个miRNA
            drug = np.random.choice(all_drugs)  # 随机选择一个药物
            if (mirna, drug) not in positive_pairs:  # 如果该miRNA-药物对不在正样本集合中
                negative_samples.append({'miRNA': mirna, 'drug': drug, 'label': 0})  # 添加到负样本列表，标签为0
                break  # 跳出while循环，继续生成下一个负样本
    
    return pd.DataFrame(negative_samples)  # 将负样本列表转换为Pandas数据框并返回


class BalancedmiRNADrugDataset(Dataset):  # 定义BalancedmiRNADrugDataset类，继承自Dataset，用于创建平衡的正负样本数据集
    def __init__(self, data_path, mirna_seq_path, drug_smiles_path,  # 初始化方法，接收数据文件路径和配置参数
                 negative_ratio=1, max_seq_len=100, is_train=True, train_ratio=0.8):  # 参数：负样本比例、最大序列长度、是否为训练集、训练集比例
        
        positive_data = pd.read_csv(data_path, header=None)  # 从CSV文件读取正样本数据，无表头
        positive_data.columns = ['miRNA', 'drug', 'pubchem_id']  # 为数据框设置列名：miRNA名称、药物名称、PubChem ID
        positive_data['label'] = 1  # 为所有正样本添加标签列，值为1
        
        all_mirnas = positive_data['miRNA'].unique()  # 获取所有唯一的miRNA名称
        all_drugs = positive_data['drug'].unique()  # 获取所有唯一的药物名称
        positive_pairs = set(zip(positive_data['miRNA'], positive_data['drug']))  # 创建正样本对的集合
        
        negative_samples = []  # 初始化负样本列表
        np.random.seed(42)  # 设置随机种子为42，确保结果可复现
        for _ in range(len(positive_data) * negative_ratio):  # 循环生成指定数量的负样本
            while True:  # 持续循环直到生成有效的负样本
                mirna = np.random.choice(all_mirnas)  # 随机选择一个miRNA
                drug = np.random.choice(all_drugs)  # 随机选择一个药物
                if (mirna, drug) not in positive_pairs:  # 如果该miRNA-药物对不在正样本集合中
                    negative_samples.append({'miRNA': mirna, 'drug': drug, 'label': 0})  # 添加到负样本列表，标签为0
                    break  # 跳出while循环，继续生成下一个负样本
        
        negative_data = pd.DataFrame(negative_samples)  # 将负样本列表转换为Pandas数据框
        
        self.data = pd.concat([positive_data, negative_data], ignore_index=True)  # 将正样本和负样本数据框合并，忽略原索引
        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)  # 随机打乱数据顺序，设置随机种子确保可复现，重置索引
        
        n_total = len(self.data)  # 获取总样本数量
        n_train = int(n_total * train_ratio)  # 计算训练集样本数量（总数量 × 训练集比例）
        
        if is_train:  # 如果是训练集
            self.data = self.data[:n_train]  # 取前n_train个样本作为训练集
        else:  # 如果是测试集/验证集
            self.data = self.data[n_train:]  # 取剩余样本作为测试集
        
        self.mirna_seq = pd.read_csv(mirna_seq_path, header=None)  # 从CSV文件读取miRNA序列数据，无表头
        self.mirna_seq.columns = ['miRNA', 'sequence']  # 为miRNA序列数据框设置列名：miRNA名称、序列字符串
        self.mirna_seq_dict = dict(zip(self.mirna_seq['miRNA'], self.mirna_seq['sequence']))  # 创建miRNA名称到序列的字典映射
        
        self.drug_smiles = pd.read_csv(drug_smiles_path, header=None)  # 从CSV文件读取药物SMILES字符串数据，无表头
        self.drug_smiles.columns = ['drug', 'smiles']  # 为药物SMILES数据框设置列名：药物名称、SMILES字符串
        self.drug_smiles_dict = dict(zip(self.drug_smiles['drug'], self.drug_smiles['smiles']))  # 创建药物名称到SMILES的字典映射
        
        self.max_seq_len = max_seq_len  # 保存最大序列长度参数
        self.nucleotide_to_idx = {'A': 1, 'U': 2, 'G': 3, 'C': 4, 'N': 0}  # 定义核苷酸到索引的映射字典
        
        self.drug_graphs = {}  # 初始化药物图字典
        self._build_drug_graphs()  # 调用内部方法构建所有药物的分子图表示
        
        self.mirna_cache = {}  # 初始化miRNA序列编码缓存字典
        
    def _build_drug_graphs(self):  # 定义内部方法，用于构建所有药物的分子图表示
        for drug, smiles in self.drug_smiles_dict.items():  # 遍历所有药物的名称和对应的SMILES字符串
            mol = Chem.MolFromSmiles(smiles)  # 使用RDKit从SMILES字符串解析分子对象
            if mol is None:  # 如果分子对象解析失败
                continue  # 跳过当前药物
            
            num_atoms = mol.GetNumAtoms()  # 获取分子中的原子数量
            adj = Chem.GetAdjacencyMatrix(mol)  # 获取分子的邻接矩阵
            adj = adj + np.eye(num_atoms)  # 在邻接矩阵上加上单位矩阵，实现自环
            adj = torch.FloatTensor(adj)  # 将邻接矩阵转换为PyTorch浮点张量
            
            node_features = []  # 初始化节点特征列表
            for atom in mol.GetAtoms():  # 遍历分子中的所有原子
                features = self._get_atom_features(atom)  # 获取当前原子的特征向量
                node_features.append(features)  # 将原子特征添加到列表中
            
            if len(node_features) > 0:  # 如果成功提取了原子特征
                node_features = torch.FloatTensor(node_features)  # 将节点特征列表转换为PyTorch浮点张量
                self.drug_graphs[drug] = (node_features, adj)  # 将药物名称映射到(节点特征, 邻接矩阵)元组
    
    def _get_atom_features(self, atom):  # 定义内部方法，用于提取单个原子的特征向量
        features = []  # 初始化特征列表
        
        atom_type = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'other']  # 定义原子类型列表
        atom_symbol = atom.GetSymbol()  # 获取原子的化学符号
        atom_feat = [0] * len(atom_type)  # 初始化原子类型特征向量
        if atom_symbol in atom_type:  # 如果原子符号在预定义列表中
            atom_feat[atom_type.index(atom_symbol)] = 1  # 将对应位置设为1
        else:  # 如果原子符号不在预定义列表中
            atom_feat[-1] = 1  # 将"other"位置设为1
        features.extend(atom_feat)  # 将原子类型特征添加到总特征列表
        
        degree = [0, 1, 2, 3, 4, 5]  # 定义原子度数的可能值列表
        d = atom.GetDegree()  # 获取当前原子的度数
        degree_feat = [0] * len(degree)  # 初始化度数特征向量
        if d < len(degree):  # 如果度数在预定义范围内
            degree_feat[d] = 1  # 将对应位置设为1
        features.extend(degree_feat)  # 将度数特征添加到总特征列表
        
        hybridization = [Chem.rdchem.HybridizationType.SP,  # 定义杂化类型列表，包含SP杂化
                        Chem.rdchem.HybridizationType.SP2,  # SP2杂化
                        Chem.rdchem.HybridizationType.SP3,  # SP3杂化
                        Chem.rdchem.HybridizationType.SP3D,  # SP3D杂化
                        Chem.rdchem.HybridizationType.SP3D2]  # SP3D2杂化
        h = atom.GetHybridization()  # 获取当前原子的杂化类型
        hybrid_feat = [0] * len(hybridization)  # 初始化杂化特征向量
        if h in hybridization:  # 如果杂化类型在预定义列表中
            hybrid_feat[hybridization.index(h)] = 1  # 将对应位置设为1
        features.extend(hybrid_feat)  # 将杂化特征添加到总特征列表
        
        features.append(atom.GetIsAromatic())  # 添加原子是否为芳香族原子的布尔值
        features.append(atom.HasQuery())  # 添加原子是否有查询属性的布尔值
        
        formal_charge = atom.GetFormalCharge()  # 获取原子的形式电荷
        features.append(formal_charge)  # 将形式电荷添加到特征列表
        
        num_hs = atom.GetTotalNumHs()  # 获取原子连接的氢原子总数
        features.append(num_hs)  # 将氢原子数量添加到特征列表
        
        features.append(atom.IsInRing())  # 添加原子是否在环结构中的布尔值
        
        features.append(atom.GetMass() / 100.0)  # 添加归一化的原子质量（除以100）
        
        features.append(atom.GetExplicitValence())  # 添加原子的显式化合价
        features.append(atom.GetImplicitValence())  # 添加原子的隐式化合价
        
        return features  # 返回包含所有特征的列表
    
    def _encode_mirna_sequence(self, sequence):  # 定义内部方法，用于对miRNA序列进行编码
        sequence = sequence.upper()  # 将序列转换为大写字母
        sequence = sequence.replace('T', 'U')  # 将T替换为U（RNA序列）
        
        encoded = [self.nucleotide_to_idx.get(nuc, 0) for nuc in sequence]  # 将每个核苷酸转换为索引值
        
        if len(encoded) < self.max_seq_len:  # 如果序列长度小于最大长度
            encoded = encoded + [0] * (self.max_seq_len - len(encoded))  # 在末尾填充0
        else:  # 如果序列长度大于或等于最大长度
            encoded = encoded[:self.max_seq_len]  # 截断序列
        
        return torch.LongTensor(encoded)  # 返回编码后的PyTorch长整型张量
    
    def __len__(self):  # 定义特殊方法，返回数据集的样本总数
        return len(self.data)  # 返回数据框的行数
    
    def __getitem__(self, idx):  # 定义特殊方法，根据索引获取单个样本
        row = self.data.iloc[idx]  # 根据索引获取数据框中的一行
        mirna_name = row['miRNA']  # 获取miRNA名称
        drug_name = row['drug']  # 获取药物名称
        label = row['label']  # 获取标签（1为正样本，0为负样本）
        
        if mirna_name in self.mirna_cache:  # 如果miRNA名称在缓存中
            mirna_seq_encoded = self.mirna_cache[mirna_name]  # 从缓存获取编码序列
        else:  # 如果miRNA名称不在缓存中
            sequence = self.mirna_seq_dict.get(mirna_name, '')  # 从字典获取序列
            mirna_seq_encoded = self._encode_mirna_sequence(sequence)  # 编码序列
            self.mirna_cache[mirna_name] = mirna_seq_encoded  # 存入缓存
        
        if drug_name not in self.drug_graphs:  # 如果药物名称不在药物图字典中
            drug_name = list(self.drug_graphs.keys())[0]  # 使用第一个可用药物替代
        
        node_features, adj = self.drug_graphs[drug_name]  # 获取节点特征和邻接矩阵
        
        return {  # 返回包含样本信息的字典
            'mirna_seq': mirna_seq_encoded,  # 编码后的miRNA序列
            'drug_node_features': node_features,  # 药物节点特征
            'drug_adj': adj,  # 药物邻接矩阵
            'label': torch.FloatTensor([label]),  # 标签张量
            'mirna_name': mirna_name,  # miRNA名称
            'drug_name': drug_name  # 药物名称
        }
