"""
GRU模型定义
支持物理特征输入、Rollout训练和Scheduled Sampling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from physics_feature_engineering import PhysicsFeatureEngineer, PhysicsFeatureConfig


class PhysicsAwareGRU(nn.Module):
    """融入物理知识的GRU模型"""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 output_dim: int = None,
                 horizon: int = 1,
                 dropout: float = 0.1,
                 physics_config: PhysicsFeatureConfig = None,
                 use_physics_features: bool = True,
                 use_domain_adaptation: bool = True):
        """
        Args:
            input_dim: 输入特征维度 (POD系数维度)
            hidden_dim: GRU隐藏层维度
            num_layers: GRU层数
            output_dim: 输出维度 (POD系数维度)
            horizon: 预测步长
            dropout: Dropout概率
            physics_config: 物理特征配置
            use_physics_features: 是否使用物理派生特征
            use_domain_adaptation: 是否使用域适应
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        self.horizon = horizon
        self.use_physics_features = use_physics_features
        self.use_domain_adaptation = use_domain_adaptation

        # 物理特征工程器
        if use_physics_features:
            self.physics_engineer = PhysicsFeatureEngineer(physics_config)
            physics_feature_dim = 5  # 标准化后的物理特征维度
        else:
            physics_feature_dim = 0

        # GRU主干网络
        self.gru = nn.GRU(
            input_size=input_dim + physics_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # 物理特征嵌入层（可选）
        if use_physics_features:
            self.physics_embedding = nn.Sequential(
                nn.Linear(physics_feature_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, physics_feature_dim)
            )

        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, self.horizon * self.output_dim)

        # 域适应相关组件
        if use_domain_adaptation:
            self.domain_classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 3),  # 3个域：轻载、中载、重载
                nn.LogSoftmax(dim=-1)
            )

        # 初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, seq: torch.Tensor, cond: torch.Tensor,
                physics_features: torch.Tensor = None, single_step: bool = False) -> torch.Tensor:
        """
        Args:
            seq: 输入序列 [B, L, input_dim]
            cond: 条件向量 [B, 2] (rpm, ambient)
            physics_features: 物理派生特征 [B, physics_feature_dim]
            single_step: 是否只预测单步（用于rollout训练）
        """
        batch_size, seq_len, _ = seq.size()

        # 准备物理特征
        if self.use_physics_features:
            if physics_features is None:
                # 实时计算物理特征
                rpm = cond[:, 0]
                ambient = cond[:, 1]
                physics_features, _ = self.physics_engineer.process_batch_physics_features(
                    rpm, ambient, variation_level='normal'
                )
                physics_features = physics_features.to(seq.device)

            # 物理特征嵌入
            physics_emb = self.physics_embedding(physics_features)  # [B, physics_feature_dim]
            # 将物理特征扩展到序列长度
            physics_emb = physics_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, physics_feature_dim]
        else:
            physics_emb = torch.zeros(batch_size, seq_len, 0, device=seq.device)

        # 拼接物理特征到输入序列
        gru_input = torch.cat([seq, physics_emb], dim=-1)  # [B, L, input_dim + physics_feature_dim]

        # GRU前向传播
        gru_output, hidden = self.gru(gru_input)

        # 取最后一个时间步的输出
        last_output = gru_output[:, -1, :]  # [B, hidden_dim]

        # 输出投影
        output = self.output_projection(last_output)  # [B, horizon * output_dim]

        if single_step:
            # 单步预测：只返回第一步预测 [B, output_dim]
            return output[:, :self.output_dim].unsqueeze(1)  # [B, 1, output_dim]
        else:
            # 多步预测：返回所有预测步长 [B, horizon, output_dim]
            return output.view(batch_size, self.horizon, self.output_dim)

    def get_domain_features(self, seq: torch.Tensor, cond: torch.Tensor,
                           physics_features: torch.Tensor = None) -> torch.Tensor:
        """获取域分类特征（用于域适应）"""
        batch_size = seq.size(0)

        # 前向传播获取隐藏特征
        if self.use_physics_features and physics_features is None:
            rpm = cond[:, 0]
            ambient = cond[:, 1]
            physics_features, _ = self.physics_engineer.process_batch_physics_features(
                rpm, ambient, variation_level='none'
            )
            physics_features = physics_features.to(seq.device)

        # 使用虚拟的零序列获取隐藏特征（只为域适应）
        dummy_seq = torch.zeros(batch_size, 1, self.input_dim, device=seq.device)
        with torch.no_grad():
            gru_output, hidden = self.gru(dummy_seq)
            domain_features = hidden[-1]  # 取最后一层的隐藏状态

        return domain_features

    def compute_domain_loss(self, domain_features: torch.Tensor,
                           domain_labels: List[str]) -> torch.Tensor:
        """计算域适应损失"""
        if not self.use_domain_adaptation:
            return torch.tensor(0.0, device=domain_features.device)

        # 将域标签转换为数字
        domain_label_map = {
            'light_thermal_load': 0,
            'medium_thermal_load': 1,
            'heavy_thermal_load': 2
        }

        domain_targets = torch.tensor(
            [domain_label_map[label] for label in domain_labels],
            device=domain_features.device,
            dtype=torch.long
        )

        # 域分类器预测
        domain_logits = self.domain_classifier(domain_features)
        domain_loss = F.nll_loss(domain_logits, domain_targets)

        return domain_loss


def append_condition(seq: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """将条件向量扩展到时间维度并拼接到输入序列上。"""
    if cond.dim() == 2:
        cond = cond.unsqueeze(1)
    cond = cond.expand(-1, seq.size(1), -1)
    return torch.cat([seq, cond], dim=-1)
