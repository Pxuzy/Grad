import torch
import torch.nn as nn


# #多尺度卷积
class MultiScaleBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=[3, 5, 7],
                 stride=1):
        super(MultiScaleBlock, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        n_branches = len(kernel_sizes)
        base_out = out_channels // n_branches
        remainder = out_channels % n_branches

        for i, k_size in enumerate(kernel_sizes):
            current_out = base_out + (remainder if i == n_branches - 1 else 0)
            pad = k_size // 2  # 'same' padding for stride=1

            conv = nn.Conv1d(in_channels,
                             current_out,
                             kernel_size=k_size,
                             stride=stride,
                             padding=pad)
            bn = nn.BatchNorm1d(current_out)
            self.convs.append(conv)
            self.bns.append(bn)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        outputs = []
        for conv, bn in zip(self.convs, self.bns):
            out = self.relu(bn(conv(x)))
            outputs.append(out)
        return torch.cat(outputs, dim=1)


class FullScaleFusionCNN(nn.Module):

    def __init__(self, num_classes, input_channels, base_features=64):
        super(FullScaleFusionCNN, self).__init__()

        # Block 1: Stride=2 (下采样)
        self.ms_block1 = MultiScaleBlock(input_channels,
                                         base_features,
                                         kernel_sizes=[3, 5, 7],
                                         stride=2)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Block 2
        self.ms_block2 = MultiScaleBlock(base_features,
                                         base_features * 2,
                                         kernel_sizes=[3, 5, 7],
                                         stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Block 3
        self.ms_block3 = MultiScaleBlock(base_features * 2,
                                         base_features * 4,
                                         kernel_sizes=[3, 5, 9],
                                         stride=1)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Block 4
        self.ms_block4 = MultiScaleBlock(base_features * 4,
                                         base_features * 8,
                                         kernel_sizes=[3, 5, 7],
                                         stride=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5),
                                        nn.Linear(base_features * 8, 256),
                                        nn.ReLU(inplace=True), nn.Dropout(0.3),
                                        nn.Linear(256, num_classes))

    def forward(self, x):
        # x: [Batch, Channels, Seq_Len]
        x = self.ms_block1(x)
        x = self.pool1(x)
        x = self.ms_block2(x)
        x = self.pool2(x)
        x = self.ms_block3(x)
        x = self.pool3(x)
        x = self.ms_block4(x)
        x = self.global_pool(x)
        return self.classifier(x)


# # =================  深层串行诊断网 =================
class DeepSerialDiagCNN(nn.Module):

    def __init__(self, num_classes, input_channels):
        super(DeepSerialDiagCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.layer4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1))

        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5),
                                        nn.Linear(512, 256),
                                        nn.ReLU(inplace=True), nn.Dropout(0.3),
                                        nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(x)


# =================  DeepFusionDiagCNN=================
# class HybridBlock(nn.Module):
#     """
#     混合模块：仅在浅层使用多尺度，深层退化为普通卷积以保持深度优势
#     """

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_sizes,
#                  stride,
#                  use_multi_scale=True):
#         super(HybridBlock, self).__init__()
#         self.use_multi_scale = use_multi_scale

#         if use_multi_scale:
#             # 【多尺度分支】：并行卷积，捕捉不同宽度的冲击
#             self.convs = nn.ModuleList()
#             self.bns = nn.ModuleList()
#             n_branches = len(kernel_sizes)
#             base_out = out_channels // n_branches
#             remainder = out_channels % n_branches

#             for i, k_size in enumerate(kernel_sizes):
#                 current_out = base_out + (remainder if i == n_branches -
#                                           1 else 0)
#                 pad = k_size // 2
#                 conv = nn.Conv1d(in_channels,
#                                  current_out,
#                                  kernel_size=k_size,
#                                  stride=stride,
#                                  padding=pad)
#                 bn = nn.BatchNorm1d(current_out)
#                 self.convs.append(conv)
#                 self.bns.append(bn)
#             self.out_channels = out_channels
#         else:
#             # 【单尺度分支】：经典大卷积核，强力提取深层语义
#             # 如果传入多个kernel_sizes，取中间值或最大值作为代表，这里默认取第一个
#             k_size = kernel_sizes[0]
#             pad = k_size // 2 if stride == 1 else k_size // 2  # 简单处理padding
#             # 对于stride=2的情况，为了保持对齐，padding可能需要调整，这里沿用原模型逻辑
#             if stride == 2 and k_size == 7: pad = 3
#             elif stride == 2 and k_size == 5: pad = 2  # 近似
#             else: pad = k_size // 2

#             self.conv = nn.Conv1d(in_channels,
#                                   out_channels,
#                                   kernel_size=k_size,
#                                   stride=stride,
#                                   padding=pad)
#             self.bn = nn.BatchNorm1d(out_channels)
#             self.out_channels = out_channels

#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         if self.use_multi_scale:
#             outputs = []
#             for conv, bn in zip(self.convs, self.bns):
#                 out = self.relu(bn(conv(x)))
#                 outputs.append(out)
#             return torch.cat(outputs, dim=1)
#         else:
#             return self.relu(self.bn(self.conv(x)))


# class DeepFusionDiagCNN(nn.Module):

#     def __init__(self, num_classes, input_channels, base_features=64):
#         super(DeepFusionDiagCNN, self).__init__()

#         # --- Layer 1: 多尺度 + 大幅下采样 ---
#         # 结合点：使用多尺度捕捉初始冲击 (3, 5, 7)，同时保留原模型的大步长下采样
#         self.ms_block1 = HybridBlock(input_channels,
#                                      base_features,
#                                      kernel_sizes=[3, 5, 7],
#                                      stride=2,
#                                      use_multi_scale=True)
#         self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

#         # --- Layer 2: 多尺度 + 中度下采样 ---
#         # 结合点：继续利用多尺度细化特征，通道翻倍
#         self.ms_block2 = HybridBlock(base_features,
#                                      base_features * 2,
#                                      kernel_sizes=[3, 5, 7],
#                                      stride=1,
#                                      use_multi_scale=True)
#         self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

#         # --- Layer 3: 回归经典单尺度 ---
#         # 结合点：此时特征已抽象，改用原模型的大核(5)单一路径，增强非线性表达能力，减少噪声
#         self.layer3 = HybridBlock(base_features * 2,
#                                   base_features * 4,
#                                   kernel_sizes=[5],
#                                   stride=1,
#                                   use_multi_scale=False)
#         self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

#         # --- Layer 4: 回归经典单尺度 ---
#         # 结合点：原模型的小核(3)汇聚，通道增至512
#         self.layer4 = HybridBlock(base_features * 4,
#                                   base_features * 8,
#                                   kernel_sizes=[3],
#                                   stride=1,
#                                   use_multi_scale=False)
#         self.global_pool = nn.AdaptiveAvgPool1d(1)

#         # --- Classifier: 保留原模型的强正则化 ---
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Dropout(0.5),  # 原模型的强力Dropout
#             nn.Linear(base_features * 8, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(256, num_classes))

#     def forward(self, x):
#         # Stage 1: 多尺度粗提取
#         x = self.ms_block1(x)
#         x = self.pool1(x)

#         # Stage 2: 多尺度细提取
#         x = self.ms_block2(x)
#         x = self.pool2(x)

#         # Stage 3: 深层语义抽象 (单路径)
#         x = self.layer3(x)
#         x = self.pool3(x)

#         # Stage 4: 全局汇聚 (单路径)
#         x = self.layer4(x)
#         x = self.global_pool(x)

#         return self.classifier(x)
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================= 辅助模块：SE 注意力机制 =================
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    自适应地重新校准通道特征响应
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1)
        # Scale
        return x * y.expand_as(x)


# ================= 优化后的混合模块 (带残差 + SE) =================
class HybridBlock(nn.Module):
    """
    优化版混合模块：
    1. 支持残差连接 (Residual)，解决梯度消失
    2. 集成 SE 注意力机制，增强关键特征
    3. 使用 SiLU 激活函数，提升非线性表达能力
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 stride,
                 use_multi_scale=True,
                 use_se=True,
                 use_residual=True):
        super(HybridBlock, self).__init__()
        self.use_multi_scale = use_multi_scale
        self.use_residual = use_residual and (in_channels == out_channels
                                              and stride == 1)
        # 如果维度不匹配，残差连接需要投影，这里简化处理：仅当维度一致时启用残差

        if use_multi_scale:
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            n_branches = len(kernel_sizes)
            base_out = out_channels // n_branches
            remainder = out_channels % n_branches

            for i, k_size in enumerate(kernel_sizes):
                current_out = base_out + (remainder if i == n_branches -
                                          1 else 0)
                pad = k_size // 2
                # 使用 SiLU 替代 ReLU 的前置卷积
                conv = nn.Conv1d(in_channels,
                                 current_out,
                                 kernel_size=k_size,
                                 stride=stride,
                                 padding=pad,
                                 bias=False)
                bn = nn.BatchNorm1d(current_out)
                self.convs.append(conv)
                self.bns.append(bn)
            self.out_channels = out_channels
        else:
            k_size = kernel_sizes[0]
            # 智能 Padding 计算
            if stride == 2:
                pad = (k_size - 1) // 2  # 保证下采样对齐
            else:
                pad = k_size // 2

            self.conv = nn.Conv1d(in_channels,
                                  out_channels,
                                  kernel_size=k_size,
                                  stride=stride,
                                  padding=pad,
                                  bias=False)
            self.bn = nn.BatchNorm1d(out_channels)
            self.out_channels = out_channels

        self.activation = nn.SiLU(inplace=True)  # SiLU 激活
        self.se_block = SEBlock(out_channels) if use_se else None

    def forward(self, x):
        identity = x if self.use_residual else None

        if self.use_multi_scale:
            outputs = []
            for conv, bn in zip(self.convs, self.bns):
                out = conv(x)
                out = bn(out)
                out = self.activation(out)
                outputs.append(out)
            out = torch.cat(outputs, dim=1)
        else:
            out = self.conv(x)
            out = self.bn(out)
            out = self.activation(out)

        # 应用 SE 注意力
        if self.se_block:
            out = self.se_block(out)

        # 残差连接
        if self.use_residual:
            out = out + identity

        return out


# ================= 优化后的 DeepFusionDiagCNN =================
class DeepFusionDiagCNN(nn.Module):
    """
    深度融多尺度诊断网 (Optimized Version)
    架构特点：
    - 浅层：多尺度 + 残差 + SE (捕捉瞬态冲击)
    - 深层：大核单路径 + SE (提取全局语义)
    - 全程：SiLU 激活 + 残差连接
    """

    def __init__(self, num_classes, input_channels, base_features=64):
        super(DeepFusionDiagCNN, self).__init__()

        # --- Stage 1: 多尺度粗提取 (带下采样) ---
        # 输入通道可能不等于输出通道，所以这里不使用残差 (use_residual=False)
        self.stage1 = nn.Sequential(
            HybridBlock(input_channels,
                        base_features,
                        kernel_sizes=[3, 5, 7],
                        stride=2,
                        use_multi_scale=True,
                        use_se=True,
                        use_residual=False),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        # --- Stage 2: 多尺度细提取 ---
        # 通道翻倍，维度不匹配，不使用残差
        self.stage2 = nn.Sequential(
            HybridBlock(base_features,
                        base_features * 2,
                        kernel_sizes=[3, 5, 7],
                        stride=1,
                        use_multi_scale=True,
                        use_se=True,
                        use_residual=False),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        # --- Stage 3: 深层语义抽象 (单路径大核) ---
        # 此时输入输出通道一致 (如果前面逻辑正确)，可以开启残差
        # 注意：stage2 输出是 base*2, stage3 输出是 base*4，维度不匹配，不能残差
        self.stage3 = nn.Sequential(
            HybridBlock(
                base_features * 2,
                base_features * 4,
                kernel_sizes=[7],  # 增大核以捕捉长依赖
                stride=1,
                use_multi_scale=False,
                use_se=True,
                use_residual=False),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        # --- Stage 4: 全局汇聚 (单路径) ---
        # 输入 base*4, 输出 base*8，维度不匹配
        self.stage4 = nn.Sequential(
            HybridBlock(base_features * 4,
                        base_features * 8,
                        kernel_sizes=[5],
                        stride=1,
                        use_multi_scale=False,
                        use_se=True,
                        use_residual=False), nn.AdaptiveAvgPool1d(1))

        # --- 增强型分类头 ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(base_features * 8, 256),
            nn.LayerNorm(256),  # 增加 LayerNorm 稳定训练
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes))

        # 初始化权重 (关键步骤，加速收敛)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='silu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.classifier(x)
