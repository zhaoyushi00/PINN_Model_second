import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset_transient import SnapshotSequenceDataset
from dataset_transient import PrecomputedCoeffDataset
from temporal_models import TCNRegressor, append_condition


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
    """基于子采样节点的瞬态热传导残差，避免 NxN 距离矩阵导致 OOM。
    归一化处理，使残差量纲统一。
    - coeff_seq: [L+1, r]
    - coords: [3, N]
    """
    device = Phi.device
    N = coords.shape[1]

    # 子采样节点
    if N > max_nodes:
        idx_sample = torch.randperm(N, device=device)[:max_nodes]
    else:
        idx_sample = torch.arange(N, device=device)

    pts = coords.t()[idx_sample]  # [n,3]
    # 距离矩阵在子集上计算（n<=max_nodes）
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
    # 残差 = ρ*c_p*dT/dt - λ*lap
    # 归一化：除以 ρ*c_p 使量纲统一到 K/s
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


@dataclass
class TrainConfig:
    snapshots_dir: str = "data/snapshots"
    meta_csv: str = "data/snapshots/meta_transient.csv"
    phi_path: str = "model/pod_phi.npy"
    mean_path: str = "model/pod_mean.npy"
    sequence_length: int = 60
    horizon: int = 1
    stride: int = 1
    batch_size: int = 2
    num_workers: int = 0
    lr: float = 1e-3
    max_epochs: int = 20
    device: str = "cuda"
    dropout: float = 0.1
    channels = (128, 128, 128)
    kernel_size: int = 3
    physics_weight: float = 1e-4
    physics_knn: int = 8
    physics_max_nodes: int = 5000
    time_step: float = 1.0
    save_dir: str = "model"
    ckpt_name: str = "transient_tcn.pth"
    log_interval: int = 10
    sensors_map_csv: str = "data/sensors_map.csv"
    real_steady_dir: str = "data/real_steady"
    sensor_loss_weight: float = 10.0
    vel_loss_weight: float = 0.1
    a0_weight: float = 0.1  
    val_ratio: float = 0.2
    seed: int = 2025
    log_metrics_csv: str = "model/training_metrics.csv"
    use_precomputed: bool = True
    coeffs_path: str = "model/pod_coeffs.npy"
    coords_path: str = "model/pod_coords.npy"


def _prepare_loaders(cfg: TrainConfig, dataset: SnapshotSequenceDataset):
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


def _eval_on_val(model, val_loader, Phi, Tbar, coords, cfg: TrainConfig):
    model.eval()
    l2 = nn.MSELoss(reduction="mean")
    coeff_mse = []
    T_true_all = []
    T_pred_all = []
    sensor_mae_list = []
    phys_res_list = []

    with torch.no_grad():
        for batch in val_loader:
            seq_a = batch["seq_a"].to(Phi.device)
            cond = batch["cond"].to(Phi.device)
            target = batch["target_a"].to(Phi.device)

            inputs = append_condition(seq_a, cond)
            preds = model(inputs)
            coeff_mse.append(float(l2(preds, target).detach().cpu()))

            # 重建最后一步温度场
            a_last = preds[:, -1]  # [B, r]
            y_true = target[:, -1]  # [B, r]
            T_pred = (Tbar + a_last @ Phi.t()).detach().cpu().numpy()  # [B, N]
            T_true = (Tbar + y_true @ Phi.t()).detach().cpu().numpy()
            T_true_all.append(T_true)
            T_pred_all.append(T_pred)

            # 传感器 MAE（如有，对horizon所有步求平均）
            if "sensor_target" in batch and coords is not None:
                pts = coords.t()
                smap = np.loadtxt(cfg.sensors_map_csv, delimiter=",", dtype=str, skiprows=1)
                if smap.ndim == 1:
                    smap = smap[None, :]
                sx = torch.tensor(smap[:, 1].astype(np.float32), device=Phi.device)
                sy = torch.tensor(smap[:, 2].astype(np.float32), device=Phi.device)
                sz = torch.tensor(smap[:, 3].astype(np.float32), device=Phi.device)
                S = sx.numel()
                sens_idx = []
                for i in range(S):
                    d = torch.cdist(pts, torch.stack([sx[i], sy[i], sz[i]]).view(1, 3))
                    sens_idx.append(int(torch.argmin(d)))
                sens_idx = torch.tensor(sens_idx, dtype=torch.long, device=Phi.device)

                # 对horizon的每一步都计算传感器MAE
                H = preds.shape[1]
                sensor_errors = []
                for h in range(H):
                    a_h = preds[:, h, :]  # [B, r]
                    T_pred_h = (Tbar + a_h @ Phi.t()).index_select(dim=1, index=sens_idx)  # [B, S]
                    y_s_h = batch["sensor_target"][:, h, :].to(Phi.device)  # [B, S]
                    mask = ~torch.isnan(y_s_h)
                    if mask.any():
                        sensor_l1 = torch.abs(T_pred_h - y_s_h)[mask].mean()
                        sensor_errors.append(float(sensor_l1.detach().cpu()))
                if sensor_errors:
                    sensor_mae_list.append(float(np.mean(sensor_errors)))

            # 物理残差（基于单样本：seq + pred最后一步），节点子采样
            if coords is not None:
                b0 = 0
                a_last_b0 = a_last[b0].unsqueeze(0)  # [1, r]
                coeff_seq_b0 = torch.cat([seq_a[b0], a_last_b0], dim=0)  # [L+1, r]
                phys = compute_physics_residual(
                    Phi=Phi, Tbar=Tbar, coeff_seq=coeff_seq_b0, coords=coords,
                    dt=cfg.time_step, knn=cfg.physics_knn, max_nodes=cfg.physics_max_nodes,
                )
                phys_res_list.append(float(phys.detach().cpu()))

    coeff_mse_val = float(np.mean(coeff_mse)) if coeff_mse else 0.0
    T_true_all = np.concatenate(T_true_all, axis=0) if T_true_all else np.zeros((1, 1))
    T_pred_all = np.concatenate(T_pred_all, axis=0) if T_pred_all else np.zeros((1, 1))
    return {
        "coeff_mse": coeff_mse_val,
        "T_r2": r2_score(T_true_all, T_pred_all),
        "T_mae": mae(T_true_all, T_pred_all),
        "T_rmse": rmse(T_true_all, T_pred_all),
        "sensor_mae": (float(np.mean(sensor_mae_list)) if sensor_mae_list else None),
        "phys_res": (float(np.mean(phys_res_list)) if phys_res_list else None),
    }


def train(cfg: TrainConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    if cfg.use_precomputed:
        dataset = PrecomputedCoeffDataset(
            meta_csv=cfg.meta_csv,
            coeffs_path=cfg.coeffs_path,
            phi_path=cfg.phi_path,
            mean_path=cfg.mean_path,
            coords_path=cfg.coords_path,
            sequence_length=cfg.sequence_length,
            horizon=cfg.horizon,
            stride=cfg.stride,
            sensors_map_csv=cfg.sensors_map_csv,
            real_steady_dir=cfg.real_steady_dir,
        )
    else:
        dataset = SnapshotSequenceDataset(
            snapshots_dir=cfg.snapshots_dir,
            meta_csv=cfg.meta_csv,
            phi_path=cfg.phi_path,
            mean_path=cfg.mean_path,
            sequence_length=cfg.sequence_length,
            horizon=cfg.horizon,
            stride=cfg.stride,
            sensors_map_csv=cfg.sensors_map_csv,
            real_steady_dir=cfg.real_steady_dir,
        )

    train_loader, val_loader = _prepare_loaders(cfg, dataset)

    model = TCNRegressor(
        input_dim=dataset.r + len(dataset.condition_cols),
        output_dim=dataset.r,
        horizon=cfg.horizon,
        channels=cfg.channels,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
    ).to(device)

    # a0 预测网络：条件 -> 初始 POD 系数
    a0_net = nn.Sequential(
        nn.Linear(2, 128), nn.ReLU(),
        nn.Linear(128, dataset.r)
    ).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.parameters()},
        {"params": a0_net.parameters(), "lr": cfg.lr}
    ], lr=cfg.lr)
    criterion = nn.MSELoss()

    Phi = torch.from_numpy(dataset.Phi).to(device)
    Tbar = torch.from_numpy(dataset.Tbar).to(device)
    coords = torch.from_numpy(dataset.coords).to(device) if getattr(dataset, 'coords', None) is not None else None

    # 准备日志文件
    os.makedirs(cfg.save_dir, exist_ok=True)
    metrics_csv = cfg.log_metrics_csv
    with open(metrics_csv, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_coeff_mse,val_T_r2,val_T_mae,val_T_rmse,val_sensor_mae,val_phys_res\n")

    for epoch in range(cfg.max_epochs):
        model.train(); a0_net.train()
        loss_meter = 0.0
        for step, batch in enumerate(train_loader):
            seq_a = batch["seq_a"].to(device)          # [B,L,r]
            cond = batch["cond"].to(device)            # [B,2]
            target = batch["target_a"].to(device)      # [B,H,r]

            inputs = append_condition(seq_a, cond)      # [B,L,r+2]
            preds = model(inputs)                       # [B,H,r]

            # 基础：多步系数监督
            mse_loss = criterion(preds, target)
            loss = mse_loss

            # 速度正则：Δa_pred 对齐 Δa_true（稳定时间常数）
            if preds.shape[1] >= 2:
                da_pred = preds[:, 1:, :] - preds[:, :-1, :]
                da_true = target[:, 1:, :] - target[:, :-1, :]
                vel_loss = criterion(da_pred, da_true)
                loss = loss + cfg.vel_loss_weight * vel_loss

            # 初始化 a0 监督
            if "is_t0" in batch:
                is_t0 = batch["is_t0"].to(device).view(-1).float()  # [B]
                if is_t0.sum() > 0:
                    a0_target_all = seq_a[:, 0, :]
                    a0_pred_all = a0_net(cond)
                    # 逐样本加权
                    per_sample = torch.mean((a0_pred_all - a0_target_all)**2, dim=1)  # [B]
                    a0_loss = (per_sample * is_t0).sum() / (is_t0.sum() + 1e-8)
                    loss = loss + cfg.a0_weight * a0_loss
            else:
                # 兼容老数据集：退化为对所有样本监督
                a0_target = seq_a[:, 0, :]
                a0_pred = a0_net(cond)
                loss = loss + cfg.a0_weight * criterion(a0_pred, a0_target)

            # 传感器监督：扩展到 horizon 每一步
            if "sensor_target" in batch and coords is not None:
                with torch.no_grad():
                    pts = coords.t()
                    smap = np.loadtxt(cfg.sensors_map_csv, delimiter=",", dtype=str, skiprows=1)
                    if smap.ndim == 1: smap = smap[None, :]
                    sx = torch.tensor(smap[:, 1].astype(np.float32), device=device)
                    sy = torch.tensor(smap[:, 2].astype(np.float32), device=device)
                    sz = torch.tensor(smap[:, 3].astype(np.float32), device=device)
                    S = sx.numel()
                    sens_idx = []
                    for i in range(S):
                        d = torch.cdist(pts, torch.stack([sx[i], sy[i], sz[i]]).view(1, 3))
                        sens_idx.append(int(torch.argmin(d)))
                    sens_idx = torch.tensor(sens_idx, dtype=torch.long, device=device)

                H = preds.shape[1]
                sensor_losses = []
                for h in range(H):
                    a_h = preds[:, h, :]
                    T_pred = Tbar + a_h @ Phi.t()                      # [B,N]
                    T_pred_s = T_pred.index_select(1, sens_idx)        # [B,S]
                    y_s = batch["sensor_target"][:, h, :].to(device)  # [B,S]
                    mask = ~torch.isnan(y_s)
                    if mask.any():
                        sensor_l1 = torch.abs(T_pred_s - y_s)[mask].mean()
                        sensor_losses.append(sensor_l1)
                if sensor_losses:
                    loss = loss + cfg.sensor_loss_weight * (sum(sensor_losses) / len(sensor_losses))

            # 物理残差（单样本、子采样）
            if cfg.physics_weight > 0 and coords is not None:
                b0 = 0
                a_last_b0 = preds[b0, -1].unsqueeze(0)
                coeff_seq_b0 = torch.cat([seq_a[b0], a_last_b0], dim=0)
                residual = compute_physics_residual(
                    Phi=Phi,
                    Tbar=Tbar,
                    coeff_seq=coeff_seq_b0,
                    coords=coords,
                    dt=cfg.time_step,
                    knn=cfg.physics_knn,
                    max_nodes=cfg.physics_max_nodes,
                )
                loss = loss + cfg.physics_weight * residual

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(a0_net.parameters()), 1.0)
            optimizer.step()

            loss_meter += float(loss.detach().cpu())

        # 验证评估
        val_metrics = _eval_on_val(model, val_loader, Phi, Tbar, coords, cfg)

        # 打印与写日志
        print(
            f"[Epoch {epoch+1}/{cfg.max_epochs}] train_loss={loss_meter/ max(1,len(train_loader)):.6f}  "
            f"val_coeff_mse={val_metrics['coeff_mse']:.6f}  "
            f"VAL(T): R2={val_metrics['T_r2']*100:.2f}%  MAE={val_metrics['T_mae']:.4f}  RMSE={val_metrics['T_rmse']:.4f}  "
            f"sensor_MAE={(val_metrics['sensor_mae'] if val_metrics['sensor_mae'] is not None else 'NA')}  "
            f"phys_res={(val_metrics['phys_res'] if val_metrics['phys_res'] is not None else 'NA')}"
        )
        with open(metrics_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch+1},{loss_meter/ max(1,len(train_loader)):.6f},{val_metrics['coeff_mse']:.6f},{val_metrics['T_r2']:.6f},{val_metrics['T_mae']:.6f},{val_metrics['T_rmse']:.6f},{val_metrics['sensor_mae'] if val_metrics['sensor_mae'] is not None else ''},{val_metrics['phys_res'] if val_metrics['phys_res'] is not None else ''}\n"
            )

        # 保存模型
        os.makedirs(cfg.save_dir, exist_ok=True)
        ckpt = {
            "state_dict": model.state_dict(),
            "a0_state_dict": a0_net.state_dict(),
            "config": cfg.__dict__,
            "r": dataset.r,
            "sequence_length": cfg.sequence_length,
            "horizon": cfg.horizon,
            "phi_path": cfg.phi_path,
            "mean_path": cfg.mean_path,
            "meta_csv": cfg.meta_csv,
            "sensors_map_csv": cfg.sensors_map_csv,
            "real_steady_dir": cfg.real_steady_dir,
        }
        torch.save(ckpt, os.path.join(cfg.save_dir, cfg.ckpt_name))
        

if __name__ == "__main__":
    train(TrainConfig())
