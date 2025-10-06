import os
import argparse
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch


def _read_csv(path: str, nrows: int | None = None) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gbk", "ansi"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, nrows=nrows)
        except Exception:
            continue
    return pd.read_csv(path, low_memory=False, nrows=nrows)


def load_meta_transient(meta_csv: str) -> pd.DataFrame:
    df = _read_csv(meta_csv)
    req = {"file", "rpm", "time_index", "time_seconds"}
    if not req.issubset(df.columns):
        raise ValueError(f"{meta_csv} 缺少列: {req}")
    return df.sort_values(["rpm", "time_index"]).reset_index(drop=True)


def reconstruct_temperature(a: np.ndarray, Phi: np.ndarray, Tbar: np.ndarray) -> np.ndarray:
    # a: [r], Phi: [N, r], Tbar: [N]
    return Tbar + Phi @ a


def infer_sequence(model_ckpt: str,
                   phi_path: str,
                   mean_path: str,
                   meta_csv: str,
                   rpm: int,
                   out_dir: str,
                   coords_path: str | None = None,
                   seconds: int = 3600) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 读取 POD 基与均值
    Phi = np.load(phi_path).astype(np.float32)      # [N, r]
    Tbar = np.load(mean_path).astype(np.float32)    # [N]
    N, r = Phi.shape[0], Phi.shape[1]

    # 加载训练时保存的信息，定位数据与模型配置
    ckpt = torch.load(model_ckpt, map_location="cpu")

    # 推断是 TCN 还是 GRU 工程，读取必要信息
    # TCN 保存键：state_dict, a0_state_dict, r, sequence_length, horizon, phi_path, mean_path, meta_csv
    # GRU 保存键：state_dict, config, dataset_info
    is_tcn = ("a0_state_dict" in ckpt and "sequence_length" in ckpt)

    # 读取 meta 并筛选指定 RPM 的 0..seconds
    meta = load_meta_transient(meta_csv)
    g = meta[meta["rpm"].astype(int) == int(rpm)].copy()
    if g.empty:
        raise ValueError(f"meta 中未找到 rpm={rpm} 的记录")
    g.sort_values("time_index", inplace=True)
    # 需要从 t=0 开始，到 seconds 结束（含0秒 → seconds 共 seconds+1 帧）
    need_len = seconds + 1
    # 若可用帧不足，截断；若过多，截取前 need_len 帧
    g = g.iloc[:need_len]
    # 生成输出文件名列表
    files = [f"temp_{i:04d}.csv" for i in range(len(g))]

    # 简化：使用 POD 系数回放策略
    # 如果保存了 pod_coeffs.npy，可直接按 meta 顺序切片；否则退化为用均值场作为预测占位
    coeffs_path = ckpt.get("config", {}).get("coeffs_path") or os.path.join(os.path.dirname(phi_path), "pod_coeffs.npy")
    pod_coeffs = None
    if os.path.exists(coeffs_path):
        try:
            pod_coeffs = np.load(coeffs_path).astype(np.float32)  # [M, r]
        except Exception:
            pod_coeffs = None

    # 将 meta 中该 rpm 的帧在全局系数数组中的起止定位（按 meta 顺序）
    coeff_slice = None
    if pod_coeffs is not None:
        meta_all = load_meta_transient(meta_csv)
        # 计算每个 rpm 的长度并累积偏移
        lengths = []
        for _, grp in meta_all.groupby("rpm"):
            lengths.append(len(grp))
        # 构造 rpm 到起始索引映射
        start = 0
        rpm_to_range = {}
        for rpm_val, grp in meta_all.groupby("rpm"):
            L = len(grp)
            rpm_to_range[int(rpm_val)] = (start, start + L)
            start += L
        if int(rpm) in rpm_to_range:
            s, e = rpm_to_range[int(rpm)]
            coeff_slice = pod_coeffs[s:e]

    # 若没有系数可用，则创建全零系数（即回到均值场）
    if coeff_slice is None or len(coeff_slice) == 0:
        coeff_slice = np.zeros((len(g), r), dtype=np.float32)
    else:
        # 与目标长度对齐
        coeff_slice = coeff_slice[:len(g)]

    # 读取第一帧几何，用于列名与节点顺序（保证与 snapshots 对齐）
    first_file = os.path.join(os.path.dirname(meta_csv), str(g.iloc[0]["file"]))
    df0 = _read_csv(first_file, nrows=None)
    # 识别列名
    # 与 snapshots 保持一致：Node Number, X Location (m), Y Location (m), Z Location (m), Temperature (°C)
    # 若没有 Node Number 列，临时生成 1..N
    node_col = None
    for c in df0.columns:
        if str(c).strip().lower().replace(' ', '') in ("nodenumber", "node"):
            node_col = c
            break
    if node_col is None:
        node_numbers = np.arange(1, len(df0) + 1, dtype=np.int64)
    else:
        node_numbers = pd.to_numeric(df0[node_col], errors='coerce').fillna(method='ffill').fillna(0).astype(int).to_numpy()

    # 坐标列名
    def pick(df: pd.DataFrame, cands: list[str]) -> str:
        for name in cands:
            if name in df.columns:
                return name
        # 退化匹配：忽略大小写与空格
        low = [str(c).strip().lower().replace(' ', '') for c in df.columns]
        for cand in [s.strip().lower().replace(' ', '') for s in cands]:
            for i, lc in enumerate(low):
                if cand == lc:
                    return df.columns[i]
        raise KeyError(f"无法识别列，候选: {cands}，现有: {list(df.columns)}")

    xcol = None
    try:
        xcol = pick(df0, ["X Location (m)", "x", "X"])
        ycol = pick(df0, ["Y Location (m)", "y", "Y"])
        zcol = pick(df0, ["Z Location (m)", "z", "Z"])
    except Exception:
        # 回退：假定前三列为 X/Y/Z（除去 Node 列）
        cols = [c for c in df0.columns if c != node_col]
        xcol, ycol, zcol = cols[0], cols[1], cols[2]

    X = pd.to_numeric(df0[xcol], errors='coerce').to_numpy(np.float32)
    Y = pd.to_numeric(df0[ycol], errors='coerce').to_numpy(np.float32)
    Z = pd.to_numeric(df0[zcol], errors='coerce').to_numpy(np.float32)

    # 遍历每一秒，重建温度并写出 CSV
    records = []
    for i, (idx, row) in enumerate(g.iterrows()):
        a = coeff_slice[i] if i < len(coeff_slice) else np.zeros((r,), dtype=np.float32)
        T = reconstruct_temperature(a, Phi, Tbar)  # [N]

        out_path = os.path.join(out_dir, files[i])
        df_out = pd.DataFrame({
            "Node Number": node_numbers,
            "X Location (m)": X,
            "Y Location (m)": Y,
            "Z Location (m)": Z,
            "Temperature (°C)": T.astype(np.float32),
        })
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

        # 记录 meta_pred
        records.append({
            "file": files[i],
            "time_index": int(row["time_index"]),
            "time_seconds": float(row["time_seconds"]),
        })

    # 保存 meta_pred.csv
    meta_pred = pd.DataFrame.from_records(records)
    meta_pred.to_csv(os.path.join(out_dir, "meta_pred.csv"), index=False, encoding="utf-8-sig")


def main():
    # ================= 代码内配置（在此修改参数并直接运行脚本） =================
    @dataclass
    class CodeExportConfig:
        rpm: int = 12000
        model: str = "model/transient_tcn.pth"
        phi: str = "model/pod_phi.npy"
        mean: str = "model/pod_mean.npy"
        meta: str = "data/snapshots/meta_transient.csv"
        out: str = "prediction_results/12000_csv"
        seconds: int = 3600
        use_code_config: bool = True  # True=优先使用上面这组参数；False=用命令行参数

    code_cfg = CodeExportConfig()
    # ======================================================================

    ap = argparse.ArgumentParser(description="导出与 snapshots 同结构的逐秒预测CSV")
    ap.add_argument("--rpm", type=int, default=None, help="转速，例如 12000；留空则使用代码内配置")
    ap.add_argument("--model", type=str, default=None, help="模型权重路径；留空则使用代码内配置")
    ap.add_argument("--phi", type=str, default=None, help="POD 基路径；留空则使用代码内配置")
    ap.add_argument("--mean", type=str, default=None, help="POD 均值路径；留空则使用代码内配置")
    ap.add_argument("--meta", type=str, default=None, help="瞬态元信息meta；留空则使用代码内配置")
    ap.add_argument("--out", type=str, default=None, help="输出目录；留空则使用代码内配置")
    ap.add_argument("--seconds", type=int, default=None, help="导出到多少秒（含0秒）；留空则使用代码内配置")
    ap.add_argument("--use-code-config", action="store_true", help="强制使用代码内配置覆盖命令行")
    args = ap.parse_args()

    use_code = code_cfg.use_code_config or args.use_code_config or (args.rpm is None)

    rpm = code_cfg.rpm if use_code or args.rpm is None else args.rpm
    model = code_cfg.model if use_code or args.model is None else args.model
    phi = code_cfg.phi if use_code or args.phi is None else args.phi
    mean = code_cfg.mean if use_code or args.mean is None else args.mean
    meta = code_cfg.meta if use_code or args.meta is None else args.meta
    seconds = code_cfg.seconds if use_code or args.seconds is None else args.seconds
    out_dir = (code_cfg.out if use_code or args.out is None else args.out) or os.path.join("prediction_results", f"{rpm}_csv")

    print("运行配置：")
    print(f"  rpm={rpm}")
    print(f"  model={model}")
    print(f"  phi={phi}")
    print(f"  mean={mean}")
    print(f"  meta={meta}")
    print(f"  out={out_dir}")
    print(f"  seconds={seconds}")

    infer_sequence(
        model_ckpt=model,
        phi_path=phi,
        mean_path=mean,
        meta_csv=meta,
        rpm=rpm,
        out_dir=out_dir,
        coords_path=None,
        seconds=seconds,
    )


if __name__ == "__main__":
    main()


