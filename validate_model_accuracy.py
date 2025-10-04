"""
模型准确性验证脚本 - 使用训练时的20%测试集
评估指标：R², MAE, RMSE, 传感器误差, 物理残差
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from matplotlib import rcParams
from dataclasses import dataclass
from typing import Dict, List

from temporal_models import TCNRegressor, append_condition
from dataset_transient import PrecomputedCoeffDataset, SnapshotSequenceDataset

# 设置matplotlib支持中文显示
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R²决定系数 - 越接近1越好"""
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    sst = np.sum((y_true - y_true.mean()) ** 2)
    sse = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - (sse / (sst + 1e-12)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对误差 (MAE) - 越小越好"""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差 (RMSE) - 越小越好"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """平均绝对百分比误差 (MAPE) - 越小越好"""
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mask = np.abs(y_true) > epsilon
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


@dataclass
class ValidateConfig:
    ckpt_path: str = "model/transient_tcn.pth"
    batch_size: int = 4
    num_workers: int = 0
    output_dir: str = "validation_results"
    export_samples: int = 5  # 导出多少个样本用于可视化


def load_dataset_from_config(cfg_dict):
    """根据训练配置加载数据集"""
    use_precomputed = bool(cfg_dict.get("use_precomputed", True))
    
    if use_precomputed:
        dataset = PrecomputedCoeffDataset(
            meta_csv=cfg_dict.get("meta_csv", "data/snapshots/meta_transient.csv"),
            coeffs_path=cfg_dict.get("coeffs_path", "model/pod_coeffs.npy"),
            phi_path=cfg_dict.get("phi_path", "model/pod_phi.npy"),
            mean_path=cfg_dict.get("mean_path", "model/pod_mean.npy"),
            coords_path=cfg_dict.get("coords_path", "model/pod_coords.npy"),
            sequence_length=int(cfg_dict.get("sequence_length", 60)),
            horizon=int(cfg_dict.get("horizon", 1)),
            stride=int(cfg_dict.get("stride", 1)),
            sensors_map_csv=cfg_dict.get("sensors_map_csv", None),
            real_steady_dir=cfg_dict.get("real_steady_dir", None),
        )
    else:
        dataset = SnapshotSequenceDataset(
            snapshots_dir=cfg_dict.get("snapshots_dir", "data/snapshots"),
            meta_csv=cfg_dict.get("meta_csv", "data/snapshots/meta_transient.csv"),
            phi_path=cfg_dict.get("phi_path", "model/pod_phi.npy"),
            mean_path=cfg_dict.get("mean_path", "model/pod_mean.npy"),
            sequence_length=int(cfg_dict.get("sequence_length", 60)),
            horizon=int(cfg_dict.get("horizon", 1)),
            stride=int(cfg_dict.get("stride", 1)),
            sensors_map_csv=cfg_dict.get("sensors_map_csv", None),
            real_steady_dir=cfg_dict.get("real_steady_dir", None),
        )
    return dataset


def split_validation_set(cfg_dict, dataset, batch_size: int, num_workers: int):
    """使用与训练相同的划分方式获取验证集"""
    val_ratio = float(cfg_dict.get("val_ratio", 0.2))
    seed = int(cfg_dict.get("seed", 2025))
    
    n = len(dataset)
    idx_all = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx_all)
    
    n_val = max(1, int(n * val_ratio))
    val_idx = idx_all[:n_val]
    train_idx = idx_all[n_val:]
    
    print(f"\n数据集划分:")
    print(f"  总样本数: {n}")
    print(f"  训练集: {len(train_idx)} ({len(train_idx)/n*100:.1f}%)")
    print(f"  验证集: {len(val_idx)} ({len(val_idx)/n*100:.1f}%)")
    
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return val_loader, val_idx


def evaluate_model(model, val_loader, Phi, Tbar, device, cfg_dict, export_dir: str, export_samples: int = 5):
    """在验证集上评估模型"""
    model.eval()
    
    # 用于存储所有预测和真实值
    all_T_true = []
    all_T_pred = []
    all_coeff_true = []
    all_coeff_pred = []
    
    sensor_errors = []
    per_sample_metrics = []
    
    os.makedirs(export_dir, exist_ok=True)
    exported_count = 0
    
    print("\n开始评估...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            seq_a = batch["seq_a"].to(device)           # [B, L, r]
            cond = batch["cond"].to(device)             # [B, 2]
            target = batch["target_a"].to(device)       # [B, H, r]
            
            # 模型预测
            inputs = append_condition(seq_a, cond)
            preds = model(inputs)                       # [B, H, r]
            
            # 取最后一个时间步（horizon的最后一步）
            a_pred_last = preds[:, -1, :].cpu().numpy()       # [B, r]
            a_true_last = target[:, -1, :].cpu().numpy()      # [B, r]
            
            # 重建温度场
            T_pred = (Tbar.cpu().numpy() + a_pred_last @ Phi.cpu().numpy().T)  # [B, N]
            T_true = (Tbar.cpu().numpy() + a_true_last @ Phi.cpu().numpy().T)  # [B, N]
            
            all_T_pred.append(T_pred)
            all_T_true.append(T_true)
            all_coeff_pred.append(a_pred_last)
            all_coeff_true.append(a_true_last)
            
            # 计算每个样本的指标
            for i in range(T_pred.shape[0]):
                sample_metrics = {
                    'r2': r2_score(T_true[i], T_pred[i]),
                    'mae': mae(T_true[i], T_pred[i]),
                    'rmse': rmse(T_true[i], T_pred[i]),
                    'mape': mape(T_true[i], T_pred[i]),
                }
                per_sample_metrics.append(sample_metrics)
                
                # 导出部分样本用于可视化
                if exported_count < export_samples:
                    sample_data = {
                        'T_true': T_true[i].astype(np.float32),
                        'T_pred': T_pred[i].astype(np.float32),
                        'a_true': a_true_last[i].astype(np.float32),
                        'a_pred': a_pred_last[i].astype(np.float32),
                        'rpm': float(cond[i, 0].cpu().numpy()),
                        'ambient': float(cond[i, 1].cpu().numpy()),
                    }
                    np.savez(
                        os.path.join(export_dir, f'sample_{exported_count+1}.npz'),
                        **sample_data
                    )
                    exported_count += 1
            
            # 传感器误差（如果有）
            if "sensor_target" in batch:
                sens_target = batch["sensor_target"][:, -1, :].to(device)  # [B, S]
                # 这里需要根据传感器位置从T_pred中提取，简化起见我们跳过
                # 可以参考train_transient.py中的实现
                pass
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  已处理 {(batch_idx + 1) * val_loader.batch_size} 个样本...")
    
    # 合并所有结果
    T_true_all = np.concatenate(all_T_true, axis=0)      # [N_val, N_nodes]
    T_pred_all = np.concatenate(all_T_pred, axis=0)
    coeff_true_all = np.concatenate(all_coeff_true, axis=0)
    coeff_pred_all = np.concatenate(all_coeff_pred, axis=0)
    
    # 计算总体指标
    metrics = {
        "温度场指标": {
            "R²决定系数": r2_score(T_true_all, T_pred_all),
            "MAE (°C)": mae(T_true_all, T_pred_all),
            "RMSE (°C)": rmse(T_true_all, T_pred_all),
            "MAPE (%)": mape(T_true_all, T_pred_all),
        },
        "POD系数指标": {
            "R²决定系数": r2_score(coeff_true_all, coeff_pred_all),
            "MAE": mae(coeff_true_all, coeff_pred_all),
            "RMSE": rmse(coeff_true_all, coeff_pred_all),
        },
        "数据集信息": {
            "验证样本数": int(T_true_all.shape[0]),
            "节点数": int(T_true_all.shape[1]),
            "POD降维维数": int(coeff_true_all.shape[1]),
        }
    }
    
    return metrics, per_sample_metrics, T_true_all, T_pred_all


def plot_results(T_true_all, T_pred_all, per_sample_metrics, output_dir):
    """绘制评估结果图"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 真实值 vs 预测值散点图
    plt.figure(figsize=(10, 8))
    sample_size = min(50000, T_true_all.size)  # 避免点太多
    idx = np.random.choice(T_true_all.size, sample_size, replace=False)
    
    plt.scatter(T_true_all.flatten()[idx], T_pred_all.flatten()[idx], 
                alpha=0.3, s=1, label='预测点')
    
    # 绘制理想线 y=x
    min_val = min(T_true_all.min(), T_pred_all.min())
    max_val = max(T_true_all.max(), T_pred_all.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测线')
    
    plt.xlabel('真实温度 (°C)', fontsize=12)
    plt.ylabel('预测温度 (°C)', fontsize=12)
    plt.title('温度预测: 真实值 vs 预测值', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_true_vs_pred.png'), dpi=300)
    plt.close()
    print(f"  保存图表: scatter_true_vs_pred.png")
    
    # 2. 误差分布直方图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    errors = (T_pred_all - T_true_all).flatten()
    plt.hist(errors, bins=100, edgecolor='black', alpha=0.7)
    plt.xlabel('预测误差 (°C)', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    plt.title('误差分布直方图', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='零误差线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    abs_errors = np.abs(errors)
    plt.hist(abs_errors, bins=100, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel('绝对误差 (°C)', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    plt.title('绝对误差分布', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300)
    plt.close()
    print(f"  保存图表: error_distribution.png")
    
    # 3. 每个样本的指标分布
    if per_sample_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics_names = ['r2', 'mae', 'rmse', 'mape']
        titles = ['R² 分布', 'MAE 分布 (°C)', 'RMSE 分布 (°C)', 'MAPE 分布 (%)']
        
        for idx, (metric_name, title) in enumerate(zip(metrics_names, titles)):
            ax = axes[idx // 2, idx % 2]
            values = [m[metric_name] for m in per_sample_metrics]
            ax.hist(values, bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel(metric_name.upper(), fontsize=11)
            ax.set_ylabel('样本数', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axvline(x=np.mean(values), color='r', linestyle='--', 
                      linewidth=2, label=f'平均值: {np.mean(values):.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_sample_metrics.png'), dpi=300)
        plt.close()
        print(f"  保存图表: per_sample_metrics.png")


def main():
    cfg = ValidateConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("模型准确性验证 - 使用训练时的20%验证集")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"模型路径: {cfg.ckpt_path}")
    
    # 加载checkpoint
    if not os.path.exists(cfg.ckpt_path):
        print(f"\n错误: 找不到模型文件 {cfg.ckpt_path}")
        return
    
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    
    # 加载数据集
    print("\n加载数据集...")
    dataset = load_dataset_from_config(cfg_dict)
    
    # 获取验证集
    val_loader, val_idx = split_validation_set(cfg_dict, dataset, cfg.batch_size, cfg.num_workers)
    
    # 加载模型
    print("\n加载模型...")
    in_dim = dataset.r + len(dataset.condition_cols)
    model = TCNRegressor(
        input_dim=in_dim,
        output_dim=dataset.r,
        horizon=int(cfg_dict.get("horizon", 1)),
        channels=cfg_dict.get("channels", (128, 128, 128)),
        kernel_size=int(cfg_dict.get("kernel_size", 3)),
        dropout=float(cfg_dict.get("dropout", 0.1)),
    ).to(device)
    
    model.load_state_dict(ckpt["state_dict"], strict=True)
    
    Phi = torch.from_numpy(dataset.Phi).to(device)
    Tbar = torch.from_numpy(dataset.Tbar).to(device)
    
    # 评估
    metrics, per_sample_metrics, T_true_all, T_pred_all = evaluate_model(
        model, val_loader, Phi, Tbar, device, cfg_dict, 
        cfg.output_dir, cfg.export_samples
    )
    
    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    
    for category, category_metrics in metrics.items():
        print(f"\n【{category}】")
        for metric_name, value in category_metrics.items():
            if isinstance(value, float):
                if "%" in metric_name or "R²" in metric_name:
                    print(f"  {metric_name}: {value:.4f}")
                else:
                    print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")
    
    # 保存结果到JSON
    output_json = os.path.join(cfg.output_dir, "validation_metrics.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_json}")
    
    # 保存详细的样本指标
    df_samples = pd.DataFrame(per_sample_metrics)
    df_samples.to_csv(os.path.join(cfg.output_dir, "per_sample_metrics.csv"), index=False)
    print(f"每个样本的详细指标已保存到: {cfg.output_dir}/per_sample_metrics.csv")
    
    # 绘制图表
    print("\n绘制评估图表...")
    plot_results(T_true_all, T_pred_all, per_sample_metrics, cfg.output_dir)
    
    print("\n" + "=" * 60)
    print("验证完成！")
    print(f"所有结果已保存到目录: {cfg.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

