import os
import json
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from temporal_models import TCNRegressor, append_condition
from dataset_transient import PrecomputedCoeffDataset, SnapshotSequenceDataset


# 评价指标

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
class TestConfig:
    ckpt_path: str = "model/transient_tcn.pth"
    batch_size: int = 2
    num_workers: int = 0
    seed: int = 2025
    export_samples: int = 3
    export_dir: str = "predict_visualization_results"


def _prepare_dataset(cfg_dict) -> Tuple[SnapshotSequenceDataset, bool]:
    # 依据训练保存的配置构建与训练一致的数据集
    use_precomputed = bool(cfg_dict.get("use_precomputed", True))
    meta_csv = cfg_dict.get("meta_csv", "data/snapshots/meta_transient.csv")
    phi_path = cfg_dict.get("phi_path", "model/pod_phi.npy")
    mean_path = cfg_dict.get("mean_path", "model/pod_mean.npy")
    sequence_length = int(cfg_dict.get("sequence_length", 60))
    horizon = int(cfg_dict.get("horizon", 1))
    stride = int(cfg_dict.get("stride", 1))
    sensors_map_csv = cfg_dict.get("sensors_map_csv", None)
    real_steady_dir = cfg_dict.get("real_steady_dir", None)

    if use_precomputed:
        coeffs_path = cfg_dict.get("coeffs_path", "model/pod_coeffs.npy")
        coords_path = cfg_dict.get("coords_path", "model/pod_coords.npy")
        ds = PrecomputedCoeffDataset(
            meta_csv=meta_csv,
            coeffs_path=coeffs_path,
            phi_path=phi_path,
            mean_path=mean_path,
            coords_path=coords_path,
            sequence_length=sequence_length,
            horizon=horizon,
            stride=stride,
            sensors_map_csv=sensors_map_csv,
            real_steady_dir=real_steady_dir,
        )
    else:
        ds = SnapshotSequenceDataset(
            snapshots_dir=cfg_dict.get("snapshots_dir", "data/snapshots"),
            meta_csv=meta_csv,
            phi_path=phi_path,
            mean_path=mean_path,
            sequence_length=sequence_length,
            horizon=horizon,
            stride=stride,
            sensors_map_csv=sensors_map_csv,
            real_steady_dir=real_steady_dir,
        )
    return ds, use_precomputed


def _split_val(cfg_dict, ds, batch_size: int, num_workers: int):
    val_ratio = float(cfg_dict.get("val_ratio", 0.2))
    seed = int(cfg_dict.get("seed", 2025))
    n = len(ds)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_ratio))
    val_idx = idx[:n_val]
    val_loader = DataLoader(Subset(ds, val_idx.tolist()), batch_size=max(1, batch_size), shuffle=False, num_workers=num_workers)
    return val_loader


def main():
    cfg = TestConfig()

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")

    # 加载 ckpt
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})

    # 数据集与加载器
    ds, _ = _prepare_dataset(cfg_dict)
    val_loader = _split_val(cfg_dict, ds, cfg.batch_size, cfg.num_workers)

    # 模型
    in_dim = ds.r + 2  # cond: rpm, ambient
    out_dim = ds.r
    channels = cfg_dict.get("channels", (128, 128, 128))
    kernel_size = int(cfg_dict.get("kernel_size", 3))
    dropout = float(cfg_dict.get("dropout", 0.1))
    horizon = int(cfg_dict.get("horizon", 1))

    model = TCNRegressor(
        input_dim=in_dim,
        output_dim=out_dim,
        horizon=horizon,
        channels=channels,
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    Phi = torch.from_numpy(ds.Phi).to(device)
    Tbar = torch.from_numpy(ds.Tbar).to(device)

    # 评估
    l2 = nn.MSELoss(reduction="mean")
    coeff_mse = []
    T_true_all = []
    T_pred_all = []

    with torch.no_grad():
        exported = 0
        os.makedirs(cfg.export_dir, exist_ok=True)

        for batch in val_loader:
            seq_a = batch["seq_a"].to(device)
            cond = batch["cond"].to(device)
            target = batch["target_a"].to(device)

            inputs = append_condition(seq_a, cond)
            preds = model(inputs)

            coeff_mse.append(float(l2(preds, target).detach().cpu()))

            a_last = preds[:, -1]  # [B, r]
            y_true = target[:, -1]
            T_pred = (Tbar + a_last @ Phi.t()).detach().cpu().numpy()
            T_true = (Tbar + y_true @ Phi.t()).detach().cpu().numpy()

            T_true_all.append(T_true)
            T_pred_all.append(T_pred)

            # 导出若干样本
            if exported < cfg.export_samples:
                for b in range(min(T_pred.shape[0], cfg.export_samples - exported)):
                    out_path = os.path.join(cfg.export_dir, f"sample_{exported+1}_pred.npy")
                    np.save(out_path, T_pred[b].astype(np.float32))
                    out_path2 = os.path.join(cfg.export_dir, f"sample_{exported+1}_true.npy")
                    np.save(out_path2, T_true[b].astype(np.float32))
                    exported += 1
                    if exported >= cfg.export_samples:
                        break

    T_true_all = np.concatenate(T_true_all, axis=0)
    T_pred_all = np.concatenate(T_pred_all, axis=0)

    metrics = {
        "coeff_mse": float(np.mean(coeff_mse)) if coeff_mse else 0.0,
        "T_r2": r2_score(T_true_all, T_pred_all),
        "T_mae": mae(T_true_all, T_pred_all),
        "T_rmse": rmse(T_true_all, T_pred_all),
        "num_val_batches": len(val_loader),
        "num_val_samples": int(T_true_all.shape[0]),
    }

    print(
        f"VAL: coeff_mse={metrics['coeff_mse']:.6f}  "
        f"T(R2)={metrics['T_r2']*100:.2f}%  MAE={metrics['T_mae']:.4f}  RMSE={metrics['T_rmse']:.4f}"
    )

    # 写入 JSON
    with open(os.path.join(cfg.export_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"保存测试指标到 {os.path.join(cfg.export_dir, 'test_metrics.json')}")


if __name__ == "__main__":
    main()
