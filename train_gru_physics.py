"""
GRU物理感知训练脚本
支持Rollout训练、Scheduled Sampling和物理特征工程
"""
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset_transient import PrecomputedCoeffDataset
from gru_models import PhysicsAwareGRU, append_condition
from physics_feature_engineering import PhysicsFeatureEngineer, PhysicsFeatureConfig
from xgboost_residual import XGBoostResidualCorrector


@dataclass
class GRUTrainConfig:
    """GRU训练配置"""
    # 数据集参数
    snapshots_dir: str = "data/snapshots"
    meta_csv: str = "data/snapshots/meta_transient.csv"
    phi_path: str = "model/pod_phi.npy"
    mean_path: str = "model/pod_mean.npy"
    sequence_length: int = 240
    horizon: int = 12
    stride: int = 1
    batch_size: int = 8
    num_workers: int = 4
    val_ratio: float = 0.2

    # 物理特征参数
    use_physics_features: bool = True
    physics_variation_level: str = 'normal'

    # 模型参数
    input_dim: int = 10  # POD系数维度（自动获取）
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    use_domain_adaptation: bool = True

    # 训练参数
    device: str = "cuda"
    lr: float = 1e-3
    max_epochs: int = 100
    warmup_epochs: int = 10
    seed: int = 2025

    # Rollout和Scheduled Sampling参数
    rollout_horizon: int = 12
    scheduled_sampling_epochs: int = 20

    # XGBoost残差纠偏参数
    use_xgboost_residual: bool = True
    residual_window_size: int = 20
    residual_update_freq: int = 10

    # 损失权重
    main_loss_weight: float = 1.0
    domain_loss_weight: float = 0.1
    physics_consistency_weight: float = 0.05

    # 保存配置
    save_dir: str = "model"
    ckpt_name: str = "gru_physics.pth"
    log_metrics_csv: str = "model/gru_training_metrics.csv"
    coeffs_path: str = "model/pod_coeffs.npy"
    coords_path: str = "model/pod_coords.npy"


def compute_physics_residual(
    Phi: torch.Tensor,
    Tbar: torch.Tensor,
    coeff_seq: torch.Tensor,
    coords: torch.Tensor,
    density: float = 7850.0,
    heat_capacity: float = 460.0,
    conductivity: float = 35.0,
    dt: float = 1.0,
    knn: int = 8,
    max_nodes: int = 5000,
) -> torch.Tensor:
    """基于子采样节点的瞬态热传导残差，避免 NxN 距离矩阵导致 OOM。"""
    device = Phi.device
    N = coords.shape[1]

    # 子采样节点
    if N > max_nodes:
        idx_sample = torch.randperm(N, device=device)[:max_nodes]
    else:
        idx_sample = torch.arange(N, device=device)

    pts = coords.t()[idx_sample]  # [n,3]
    dist2 = torch.cdist(pts, pts, p=2) ** 2  # [n,n]
    k = min(knn, max(1, pts.size(0) - 1))
    nn_idx = torch.topk(dist2, k=k + 1, largest=False).indices[:, 1:]  # [n,k]

    temps = coeff_seq @ Phi.t() + Tbar  # [L+1, N]
    T_sub_seq = temps[:, idx_sample]    # [L+1, n]

    dT_dt = (T_sub_seq[1:] - T_sub_seq[:-1]) / dt  # [L, n]

    lap_list = []
    for t in range(T_sub_seq.size(0)):
        T_t = T_sub_seq[t]                    # [n]
        T_nn = T_t[nn_idx]                    # [n,k]
        lap = (T_nn - T_t.unsqueeze(1)).mean(dim=1)  # [n]
        lap_list.append(lap)
    lap_seq = torch.stack(lap_list)           # [L+1, n]

    # 热传导方程：ρ*c_p*∂T/∂t = λ*∇²T
    alpha = conductivity / (density * heat_capacity)  # 热扩散率 [m²/s]
    residual = dT_dt - alpha * lap_seq[:-1]  # [L, n]，单位：K/s

    return (residual ** 2).mean()


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    sst = np.sum((y_true - y_true.mean()) ** 2)
    sse = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - (sse / (sst + 1e-12)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _prepare_loaders(cfg: GRUTrainConfig, dataset: PrecomputedCoeffDataset):
    """准备训练和验证数据加载器"""
    n = len(dataset)
    idx_all = np.arange(n)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(idx_all)
    n_val = max(1, int(n * cfg.val_ratio))
    val_idx = idx_all[:n_val]
    tr_idx = idx_all[n_val:]
    train_loader = DataLoader(Subset(dataset, tr_idx.tolist()), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(Subset(dataset, val_idx.tolist()), batch_size=max(1, cfg.batch_size), shuffle=False, num_workers=cfg.num_workers)
    return train_loader, val_loader


def _eval_on_val(model, val_loader, Phi, Tbar, coords, cfg: GRUTrainConfig, physics_engineer=None):
    """验证集评估"""
    model.eval()
    l2 = nn.MSELoss(reduction="mean")
    coeff_mse = []
    T_true_all = []
    T_pred_all = []

    with torch.no_grad():
        for batch in val_loader:
            seq_a = batch["seq_a"].to(Phi.device)
            cond = batch["cond"].to(Phi.device)
            target = batch["target_a"].to(Phi.device)

            # 物理特征
            physics_features = None
            if cfg.use_physics_features and 'physics_features' in batch:
                physics_features = batch['physics_features'].to(Phi.device)

            # 模型预测（多步预测）
            preds = model(seq_a, cond, physics_features, single_step=False)

            coeff_mse.append(float(l2(preds, target).detach().cpu()))

            # 重建温度场（使用最后一步预测）
            a_last = preds[:, -1]  # [B, r]
            y_true = target[:, -1]  # [B, r]
            T_pred = (Tbar + a_last @ Phi.t()).detach().cpu().numpy()  # [B, N]
            T_true = (Tbar + y_true @ Phi.t()).detach().cpu().numpy()
            T_true_all.append(T_true)
            T_pred_all.append(T_pred)

    coeff_mse_val = float(np.mean(coeff_mse)) if coeff_mse else 0.0
    T_true_all = np.concatenate(T_true_all, axis=0) if T_true_all else np.zeros((1, 1))
    T_pred_all = np.concatenate(T_pred_all, axis=0) if T_pred_all else np.zeros((1, 1))

    return {
        "coeff_mse": coeff_mse_val,
        "T_r2": r2_score(T_true_all, T_pred_all),
        "T_mae": mae(T_true_all, T_pred_all),
        "T_rmse": rmse(T_true_all, T_pred_all),
    }


def train(cfg: GRUTrainConfig):
    """GRU物理感知训练主函数"""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 加载数据集
    dataset = PrecomputedCoeffDataset(
        meta_csv=cfg.meta_csv,
        coeffs_path=cfg.coeffs_path,
        phi_path=cfg.phi_path,
        mean_path=cfg.mean_path,
        coords_path=cfg.coords_path,
        sequence_length=cfg.sequence_length,
        horizon=cfg.horizon,
        stride=cfg.stride,
    )

    # 更新配置中的input_dim
    cfg.input_dim = dataset.r

    # 准备物理特征工程器
    physics_engineer = None
    if cfg.use_physics_features:
        physics_config = PhysicsFeatureConfig()
        physics_engineer = PhysicsFeatureEngineer(physics_config)

    train_loader, val_loader = _prepare_loaders(cfg, dataset)

    # 初始化模型
    model = PhysicsAwareGRU(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        output_dim=cfg.input_dim,
        horizon=cfg.rollout_horizon,
        dropout=cfg.dropout,
        use_physics_features=cfg.use_physics_features,
        use_domain_adaptation=cfg.use_domain_adaptation,
        physics_config=physics_config if cfg.use_physics_features else None,
    ).to(device)

    # 初始化XGBoost残差纠偏器
    residual_corrector = None
    if cfg.use_xgboost_residual:
        residual_corrector = XGBoostResidualCorrector(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            residual_window_size=cfg.residual_window_size
        )

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # 损失函数
    criterion = nn.MSELoss()

    # 加载POD基底
    Phi = torch.from_numpy(dataset.Phi).to(device)
    Tbar = torch.from_numpy(dataset.Tbar).to(device)
    coords = torch.from_numpy(dataset.coords).to(device) if hasattr(dataset, 'coords') and dataset.coords is not None else None

    # Scheduled Sampling参数
    teacher_forcing_ratio = 1.0
    ss_decay_rate = 1.0 / cfg.scheduled_sampling_epochs

    # 准备日志文件
    os.makedirs(cfg.save_dir, exist_ok=True)
    metrics_csv = cfg.log_metrics_csv
    with open(metrics_csv, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,main_loss,domain_loss,val_coeff_mse,val_T_r2,val_T_mae,val_T_rmse,teacher_forcing_ratio\n")

    print(f"开始训练GRU物理感知模型 - 输入维度: {cfg.input_dim}, 序列长度: {cfg.sequence_length}, 预测步长: {cfg.horizon}")

    for epoch in range(cfg.max_epochs):
        model.train()
        loss_meter = 0.0
        main_loss_total = 0.0
        domain_loss_total = 0.0

        for step, batch in enumerate(train_loader):
            # 准备数据
            seq_a = batch["seq_a"].to(device)          # [B,L,r]
            cond = batch["cond"].to(device)            # [B,2]
            target = batch["target_a"].to(device)      # [B,H,r]

            # 物理特征
            physics_features = None
            domain_labels = None
            if cfg.use_physics_features and 'physics_features' in batch:
                physics_features = batch['physics_features'].to(device)
                domain_labels = batch.get('domain_label', None)

            # Scheduled Sampling：决定是否使用教师强制
            use_teacher_forcing = (
                torch.rand(1).item() < teacher_forcing_ratio or
                epoch < cfg.scheduled_sampling_epochs // 2
            )

            # Rollout训练：多步预测
            rollout_losses = []
            batch_size = seq_a.size(0)

            for t in range(cfg.rollout_horizon):
                if t == 0:
                    # 第一步：使用完整序列预测
                    current_pred = model(seq_a, cond, physics_features, single_step=True)  # [B, 1, output_dim]
                    current_pred_single = current_pred.squeeze(1)  # [B, output_dim]
                else:
                    # 后续步：使用前一步预测作为输入（Rollout）
                    if use_teacher_forcing:
                        # 教师强制：使用真实的历史数据
                        current_input = seq_a[:, -1:, :]
                    else:
                        # 自回归：使用前一步预测
                        current_input = prev_pred.unsqueeze(1)

                    # 单步预测
                    current_pred = model(current_input, cond, physics_features, single_step=True)  # [B, 1, output_dim]
                    current_pred_single = current_pred.squeeze(1)  # [B, output_dim]

                # 计算单步损失
                step_loss = criterion(current_pred_single, target[:, t])
                rollout_losses.append(step_loss)
                prev_pred = current_pred_single

            # 主损失：Rollout平均损失
            main_loss = torch.stack(rollout_losses).mean()

            # 域适应损失
            domain_loss = torch.tensor(0.0, device=main_loss.device)
            if cfg.use_domain_adaptation and domain_labels is not None:
                domain_features = model.get_domain_features(seq_a, cond, physics_features)
                domain_loss = model.compute_domain_loss(domain_features, domain_labels)

            # 总损失
            total_loss = (
                cfg.main_loss_weight * main_loss +
                cfg.domain_loss_weight * domain_loss
            )

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_meter += float(total_loss.detach().cpu())
            main_loss_total += float(main_loss.detach().cpu())
            domain_loss_total += float(domain_loss.detach().cpu())

        # 验证评估
        val_metrics = _eval_on_val(model, val_loader, Phi, Tbar, coords, cfg, physics_engineer)

        # 更新Scheduled Sampling比例
        if epoch >= cfg.scheduled_sampling_epochs:
            teacher_forcing_ratio = max(0.0, teacher_forcing_ratio - ss_decay_rate)

        # 打印进度
        print(
            f"[Epoch {epoch+1}/{cfg.max_epochs}] "
            f"train_loss={loss_meter/max(1,len(train_loader)):.6f} "
            f"main_loss={main_loss_total/max(1,len(train_loader)):.6f} "
            f"domain_loss={domain_loss_total/max(1,len(train_loader)):.6f} "
            f"TF_ratio={teacher_forcing_ratio:.3f}"
        )
        print(
            f"  VAL: coeff_mse={val_metrics['coeff_mse']:.6f} "
            f"T(R2)={val_metrics['T_r2']*100:.2f}% MAE={val_metrics['T_mae']:.4f} RMSE={val_metrics['T_rmse']:.4f}"
        )

        # 记录日志
        with open(metrics_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch+1},{loss_meter/max(1,len(train_loader)):.6f},{main_loss_total/max(1,len(train_loader)):.6f},"
                f"{domain_loss_total/max(1,len(train_loader)):.6f},{val_metrics['coeff_mse']:.6f},{val_metrics['T_r2']:.6f},"
                f"{val_metrics['T_mae']:.6f},{val_metrics['T_rmse']:.6f},{teacher_forcing_ratio:.6f}\n"
            )

        # 学习率调度
        scheduler.step()

        # 保存模型
        if (epoch + 1) % 10 == 0:
            ckpt = {
                "state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "epoch": epoch + 1,
                "physics_engineer": physics_engineer,
                "residual_corrector": residual_corrector,
            }
            torch.save(ckpt, os.path.join(cfg.save_dir, f"{cfg.ckpt_name.replace('.pth', f'_epoch_{epoch+1}.pth')}"))

    # 保存最终模型
    final_ckpt = {
        "state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "physics_engineer": physics_engineer,
        "residual_corrector": residual_corrector,
        "dataset_info": {
            "r": dataset.r,
            "phi_path": cfg.phi_path,
            "mean_path": cfg.mean_path,
            "coords_path": cfg.coords_path,
        }
    }
    torch.save(final_ckpt, os.path.join(cfg.save_dir, cfg.ckpt_name))

    print(f"\nGRU物理感知训练完成！最终模型保存到: {os.path.join(cfg.save_dir, cfg.ckpt_name)}")


if __name__ == "__main__":
    # 默认配置
    cfg = GRUTrainConfig()
    train(cfg)
