import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.model_selection import train_test_split
# 数据组件
class WindTurbineDataset(Dataset):
    def __init__(self, signals, features, labels):
        self.signals = torch.FloatTensor(signals)       # (N, 3, L)
        self.features = torch.FloatTensor(features)     # (N, 3, F)
        self.labels = torch.LongTensor(labels)          # (N,)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # 返回字典格式，方便模型接收多输入
        return {
            'signal': self.signals[idx],      # (3, L)
            'feature': self.features[idx],    # (3, F)
            'label': self.labels[idx]         # scalar
        }
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

# ==========================================
# 1. 数据组件 (保持兼容你的数据结构)
# ==========================================
class WindTurbineDataset(Dataset):
    def __init__(self, signals, features, labels):
        """
        signals: (N, 3, L)
        features: (N, 3, 12)  <-- 注意：这里包含 11 维统计量 + 1 维 ID
        labels: (N,)
        """
        self.signals = torch.FloatTensor(signals)
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # 返回字典，feature 依然包含 ID 列，让模型内部去拆分
        return {
            'signal': self.signals[idx],      # (3, L)
            'feature': self.features[idx],    # (3, 12) [11 stats + 1 ID]
            'label': self.labels[idx]         # scalar
        }

# ==========================================
# 2. 编码器组件 (保持不变)
# ==========================================
class DilatedTimeEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_layers=5):
        super().__init__()
        layers = []
        # 第一层
        layers.append(nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3))
        layers.append(nn.BatchNorm1d(base_channels))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(2))
        
        # 空洞卷积层
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=dilation, dilation=dilation))
            layers.append(nn.BatchNorm1d(base_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            
        self.network = nn.Sequential(*layers)
        self.out_dim = base_channels 
        
    def forward(self, x):
        x = self.network(x)
        return torch.mean(x, dim=-1)

class SpectralEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        self.cnn2d = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) 
        )
        self.out_dim = base_channels * 4

    def forward(self, x):
        B, C, L = x.shape
        x_stft_list = []
        # 优化：利用 batch 运算加速 STFT (如果显存允许)，或者保持循环
        for b in range(B):
            batch_specs = []
            for c in range(C):
                spec = torch.stft(x[b, c], n_fft=256, hop_length=128, window=torch.hann_window(256).to(x.device), return_complex=True)
                spec_mag = torch.abs(spec)
                batch_specs.append(spec_mag)
            x_stft_list.append(torch.stack(batch_specs))
        
        x_spectrogram = torch.stack(x_stft_list) 
        x_feat = self.cnn2d(x_spectrogram)
        return x_feat.squeeze(-1).squeeze(-1)

class StatEncoder(nn.Module):
    """
    统计特征编码器
    输入: (B, 3, 11) -> 输出: (B, d_model)
    注意：这里只处理纯统计量，不包含 ID
    """
    def __init__(self, input_dim=11, channels=3, embed_dim=128):
        super().__init__()
        self.channel_mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(32 * channels, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.out_dim = embed_dim

    def forward(self, x):
        # x: (B, 3, 11)
        B, C, D = x.shape
        x = x.view(B * C, D)
        x = self.channel_mlp(x)
        x = x.view(B, -1)
        return self.fusion_mlp(x)

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        
        self.q_time = nn.Linear(dim, dim)
        self.k_freq = nn.Linear(dim, dim)
        self.v_freq = nn.Linear(dim, dim)
        
        self.q_freq = nn.Linear(dim, dim)
        self.k_time = nn.Linear(dim, dim)
        self.v_time = nn.Linear(dim, dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, feat_time, feat_freq):
        q_t = self.q_time(feat_time).unsqueeze(1)
        k_f = self.k_freq(feat_freq).unsqueeze(1)
        v_f = self.v_freq(feat_freq).unsqueeze(1)
        
        q_f = self.q_freq(feat_freq).unsqueeze(1)
        k_t = self.k_time(feat_time).unsqueeze(1)
        v_t = self.v_time(feat_time).unsqueeze(1)
        
        attn_tf = (q_t @ k_f.transpose(-2, -1)) * self.scale
        attn_tf = F.softmax(attn_tf, dim=-1)
        out_tf = (attn_tf @ v_f).squeeze(1)
        
        attn_ft = (q_f @ k_t.transpose(-2, -1)) * self.scale
        attn_ft = F.softmax(attn_ft, dim=-1)
        out_ft = (attn_ft @ v_t).squeeze(1)
        
        fused = feat_time + feat_freq + out_tf + out_ft
        fused = self.norm1(fused)
        
        out = self.ffn(fused)
        out = self.norm3(fused + out)
        
        return out

# ==========================================
# 3. 主模型 (核心修改部分)
# ==========================================
class TurbineFaultNet(nn.Module):
    def __init__(self, num_classes=2, stat_total_dim=12, d_model=32,num_turbines=8):
        """
        参数说明:
        - stat_total_dim: 输入的统计特征总维度 (11 个统计量 + 1 个 ID = 12)
        - num_turbines: 风机总数 (Embedding 表的大小)
        """
        super().__init__()
        
        # 1. 分支编码器
        self.time_encoder = DilatedTimeEncoder(in_channels=3, base_channels=64)
        self.freq_encoder = SpectralEncoder(in_channels=3, base_channels=32)
        
        # 【关键修改】StatEncoder 只接收前 11 维 (total_dim - 1)
        self.stat_encoder = StatEncoder(
            input_dim=stat_total_dim - 1, 
            channels=3, 
            embed_dim=d_model
        ) 
        
        # 2. 风机 ID 嵌入层
        # 专门处理最后一列的 ID
        self.turbine_embedding = nn.Embedding(num_embeddings=num_turbines, embedding_dim=d_model)
        
        # 投影层
        self.proj_time = nn.Linear(self.time_encoder.out_dim, d_model)
        self.proj_freq = nn.Linear(self.freq_encoder.out_dim, d_model)
        
        # 交叉注意力融合
        self.cross_attn = CrossAttentionFusion(dim=d_model, num_heads=4)
        
        # 3. 最终分类头
        # 输入 = 时频融合(d_model) + 统计特征(d_model) + ID 嵌入(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, signal, feature, turbine_id=None):
        """
        输入说明:
        - signal: (B, 3, L)
        - feature: (B, 3, 12)  <-- 最后一列是 ID
        - turbine_id: 可选，如果传入则忽略 feature 中的 ID 列，否则自动从 feature 提取
        """
        B, C, total_dim = feature.shape
        
        # --- 【核心逻辑】拆分统计特征和 ID ---
        if turbine_id is None:
            # 如果没有单独传入 ID，则从 feature 的最后一列提取
            # 假设所有通道的 ID 列数值相同，取第一个通道 [:, 0, -1]
            turbine_id = feature[:, 0, -1].long() 
            
            # 提取前 11 列作为纯统计特征
            feature_stat = feature[:, :, :-1] 
        else:
            # 如果单独传入了 ID，依然要切除 feature 的最后一列，防止维度不匹配
            feature_stat = feature[:, :, :-1]
        
        # 1. 提取时域特征
        feat_time_raw = self.time_encoder(signal)
        feat_time = self.proj_time(feat_time_raw)
        
        # 2. 提取频域特征
        feat_freq_raw = self.freq_encoder(signal)
        feat_freq = self.proj_freq(feat_freq_raw)
        
        # 3. 交叉注意力融合 (时域 <-> 频域)
        feat_fused_dynamic = self.cross_attn(feat_time, feat_freq)
        
        # 4. 编码纯统计特征 (输入形状: B, 3, 11)
        feat_stat = self.stat_encoder(feature_stat)
        
        # 5. 获取风机 ID 嵌入
        # turbine_id 形状: (B,) -> Embedding -> (B, d_model)
        # 确保 ID 在合法范围内 [0, num_turbines-1]
        turbine_id = torch.clamp(turbine_id, 0, self.turbine_embedding.num_embeddings - 1)
        feat_id = self.turbine_embedding(turbine_id)
        
        # 6. 最终融合
        final_repr = torch.cat([feat_fused_dynamic, feat_stat, feat_id], dim=1)
        
        logits = self.classifier(final_repr)
        return logits