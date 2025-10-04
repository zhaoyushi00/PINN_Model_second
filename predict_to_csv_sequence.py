import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from temporal_models import TCNRegressor, append_condition

DEFAULT_CKPT = "model/transient_tcn.pth"
DEFAULT_RPM = 9000
DEFAULT_WARMUP = 60
DEFAULT_SEQ_LEN = 60
DEFAULT_SAVE_DIR = "predict_visualization_results"
DEFAULT_T_LEN = 3600
DEFAULT_SAVE_STRIDE = 1  # 每多少秒保存一帧（1=每秒都保存）


def _read_csv(path: str):
    for enc in ("utf-8", "utf-8-sig", "gbk", "ansi"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(path, low_memory=False)


def load_ckpt_and_data(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})

    Phi = np.load(cfg.get("phi_path", "model/pod_phi.npy")).astype(np.float32)  # [N, r]
    Tbar = np.load(cfg.get("mean_path", "model/pod_mean.npy")).astype(np.float32)  # [N]
    coeffs = np.load(cfg.get("coeffs_path", "model/pod_coeffs.npy")).astype(np.float32)  # [M, r]

    meta_csv = cfg.get("meta_csv", "data/snapshots/meta_transient.csv")
    meta = _read_csv(meta_csv)
    summary_csv = os.path.join(os.path.dirname(cfg.get("phi_path", "model/pod_phi.npy")), "pod_summary.csv")
    summary = _read_csv(summary_csv)

    r = int(ckpt.get("r", coeffs.shape[1]))
    in_dim = int(r + 2)
    out_dim = r
    channels = cfg.get("channels", (128, 128, 128))
    kernel_size = int(cfg.get("kernel_size", 3))
    dropout = float(cfg.get("dropout", 0.1))
    horizon = int(cfg.get("horizon", 1))

    model = TCNRegressor(
        input_dim=in_dim,
        output_dim=out_dim,
        horizon=horizon,
        channels=channels,
        kernel_size=kernel_size,
        dropout=dropout,
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    a0_net = None
    if "a0_state_dict" in ckpt:
        a0_net = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, r)
        )
        a0_net.load_state_dict(ckpt["a0_state_dict"], strict=True)
        a0_net.eval()

    coords = np.load("model/pod_coords.npy").astype(np.float32)  # [3,N]

    return model, a0_net, cfg, Phi, Tbar, coeffs, meta, summary, coords


def select_indices(meta: pd.DataFrame, summary: pd.DataFrame, rpm: int):
    meta_map = {str(row["file"]): int(row["rpm"]) for _, row in meta.iterrows()}
    files_order = summary["file"].astype(str).tolist()
    idx = [i for i, f in enumerate(files_order) if meta_map.get(f, None) == int(rpm)]
    return np.array(idx, dtype=np.int64)


def nearest_rpm(meta: pd.DataFrame, target_rpm: int) -> int:
    rpms = sorted(set(int(x) for x in meta["rpm"].tolist()))
    if not rpms:
        raise RuntimeError("meta 中没有 rpm 列或为空")
    return min(rpms, key=lambda x: abs(x - int(target_rpm)))


def predict_sequence(model: nn.Module,
                     a0_net: nn.Module | None,
                     Phi: np.ndarray,
                     Tbar: np.ndarray,
                     coeffs: np.ndarray,
                     meta: pd.DataFrame,
                     summary: pd.DataFrame,
                     rpm: int,
                     warmup: int,
                     seq_len: int,
                     total_len: int) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if a0_net is not None:
        a0_net = a0_net.to(device)

    idx = select_indices(meta, summary, rpm)

    ambient = None
    if ("rpm" in meta.columns) and (meta["rpm"] == int(rpm)).any():
        ambient = float(meta[meta["rpm"] == int(rpm)]["ambient"].iloc[0]) if "ambient" in meta.columns else None
    else:
        near = nearest_rpm(meta, rpm)
        ambient = float(meta[meta["rpm"] == int(near)]["ambient"].iloc[0]) if "ambient" in meta.columns else None
    if ambient is None:
        ambient = 22.9
    cond = torch.tensor([rpm, ambient], dtype=torch.float32, device=device).unsqueeze(0)

    if idx.size > 0:
        T_len = idx.shape[0]
    else:
        T_len = int(total_len)

    r = Phi.shape[1]
    a_pred = np.zeros((T_len, r), dtype=np.float32)

    if a0_net is not None:
        with torch.no_grad():
            a0 = a0_net(cond).squeeze(0).detach().cpu().numpy()
        a_pred[0] = a0
        with torch.no_grad():
            for t in range(1, min(warmup, T_len)):
                start = max(0, t - seq_len)
                seq_window = torch.from_numpy(a_pred[start:t]).unsqueeze(0).to(device)
                out = model(append_condition(seq_window, cond))
                a_next = out[:, -1, :].squeeze(0).detach().cpu().numpy()
                a_pred[t] = a_next
    else:
        if idx.size == 0:
            raise RuntimeError("目标 rpm 不在 meta，且 ckpt 无 a0_net，无法初始化")
        a_truth = coeffs[idx]
        a_pred[:min(warmup, T_len)] = a_truth[:min(warmup, T_len)]

    with torch.no_grad():
        for t in range(min(warmup, T_len), T_len):
            start = max(0, t - seq_len)
            seq_window = torch.from_numpy(a_pred[start:t]).unsqueeze(0).to(device)
            out = model(append_condition(seq_window, cond))
            a_next = out[:, -1, :].squeeze(0).detach().cpu().numpy()
            a_pred[t] = a_next

    return a_pred


def save_csv_frames(a_pred: np.ndarray,
                     Phi: np.ndarray,
                     Tbar: np.ndarray,
                     coords: np.ndarray,
                     out_dir: str,
                     rpm: int,
                     ambient: float,
                     save_stride: int = 1):
    os.makedirs(out_dir, exist_ok=True)
    Phi_t = torch.from_numpy(Phi)
    Tbar_t = torch.from_numpy(Tbar)

    x, y, z = coords[0], coords[1], coords[2]
    node = np.arange(1, coords.shape[1] + 1, dtype=np.int64)

    meta_rows = []
    for t in range(0, a_pred.shape[0], max(1, int(save_stride))):
        a_t = torch.from_numpy(a_pred[t])
        T_t = (Tbar_t + Phi_t @ a_t).numpy().astype(np.float32)
        df = pd.DataFrame({
            "Node Number": node,
            "X Location (m)": x,
            "Y Location (m)": y,
            "Z Location (m)": z,
            "Temperature (°C)": T_t,
        })
        fn = f"temp_{t+1:04d}.csv"
        fpath = os.path.join(out_dir, fn)
        df.to_csv(fpath, index=False, encoding="utf-8-sig")
        meta_rows.append({"file": fn, "rpm": rpm, "ambient": ambient, "time_index": t, "time_seconds": t})

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(os.path.join(out_dir, "meta_pred.csv"), index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--rpm", type=int, default=DEFAULT_RPM)
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--t-len", type=int, default=DEFAULT_T_LEN)
    ap.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    ap.add_argument("--save-stride", type=int, default=DEFAULT_SAVE_STRIDE)
    args = ap.parse_args()

    model, a0_net, cfg, Phi, Tbar, coeffs, meta, summary, coords = load_ckpt_and_data(args.ckpt)

    # ambient 获取
    if ("rpm" in meta.columns) and (meta["rpm"] == int(args.rpm)).any():
        ambient = float(meta[meta["rpm"] == int(args.rpm)]["ambient"].iloc[0]) if "ambient" in meta.columns else 22.9
    else:
        ambient = 22.9

    a_pred = predict_sequence(
        model=model,
        a0_net=a0_net,
        Phi=Phi,
        Tbar=Tbar,
        coeffs=coeffs,
        meta=meta,
        summary=summary,
        rpm=args.rpm,
        warmup=args.warmup,
        seq_len=args.seq_len,
        total_len=args.t_len,
    )

    out_dir = os.path.join(args.save_dir, f"{int(args.rpm)}_csv")
    save_csv_frames(
        a_pred=a_pred,
        Phi=Phi,
        Tbar=Tbar,
        coords=coords,
        out_dir=out_dir,
        rpm=int(args.rpm),
        ambient=ambient,
        save_stride=args.save_stride,
    )
    print(f"已保存逐秒 CSV 到：{out_dir}")


if __name__ == "__main__":
    main()
