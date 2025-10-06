# build_pod.py
import os
import re
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd


def _read_csv(path: str, nrows: int | None = None) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gbk", "ansi"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, low_memory=False, nrows=nrows, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise last_err

# ============ 名称规范化 & 匹配工具 ============

def _norm(s: str) -> str:
    """归一化列名：小写、去空格/下划线/连字符/全角括号/单位标记等（保留字母与中文）"""
    s = str(s).strip().lower()
    # 全角括号 → 半角
    s = s.replace('（', '(').replace('）', ')').replace('[', '(').replace(']', ')')
    # 去掉常见分隔/标点
    for ch in [' ', '\u3000', '_', '-', '/', '\\', '.', ',', ':', ';', '%']:
        s = s.replace(ch, '')
    # 去掉单位标注（仅用于匹配，不改变数据单位）
    s = s.replace('(mm)', '').replace('(m)', '')
    s = s.replace('mm', '').replace('m', '')
    return s

_token_re_cache = {}
def _has_token(name_norm: str, token: str) -> bool:
    """
    在归一化后的字符串中，匹配“独立 token”的 x/y/z/temp 等。
    例如 'index' 不应匹配 'x'（因为两侧都是字母）。
    """
    key = (token,)
    if key not in _token_re_cache:
        _token_re_cache[key] = re.compile(rf'(^|[^a-z]){re.escape(token)}([^a-z]|$)')
    return bool(_token_re_cache[key].search(name_norm))

def _is_index_like(series: pd.Series) -> bool:
    """整数单调递增、步长为1 → 认为是索引列"""
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty:
        return False
    if not pd.api.types.is_integer_dtype(s):
        return False
    if not s.is_monotonic_increasing:
        return False
    d = s.diff().dropna()
    return (len(d) > 0) and (d == 1).all()

def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """尽量把能转成数值的列都转成数值"""
    df2 = df.copy()
    for c in df2.columns:
        try:
            num = pd.to_numeric(df2[c], errors='coerce')
            # 如果至少一半是有效数值，就采用
            if num.notna().mean() > 0.5:
                df2[c] = num
        except Exception:
            pass
    return df2

# ============ 列选择逻辑（强鲁棒） ============

def _pick_cols(df: pd.DataFrame):
    """
    返回: (xcol, ycol, zcol, tcol, rcol)
    识别顺序：
      1) 直接用列名匹配（x/y/z/temp/region 的丰富别名）
      2) 若温度列找不到 → 在数值列里启发式判定（优先包含 temp/t/温度；否则取最右侧数值列）
      3) 若坐标列找不到 → 在数值列里用“方差最大的三列”（排除温度列/索引列）作为 x/y/z
    """
    raw = list(df.columns)
    low = [_norm(c) for c in raw]

    # region
    rcol = None
    for orig, n in zip(raw, low):
        if n in ('region','区域','part','body','reg'):
            rcol = orig
            break

    # 温度候选（名字匹配）
    t_candidates = []
    for orig, n in zip(raw, low):
        if any(k in n for k in ['temperature', 'temp', '温度', 't_pred']):
            t_candidates.append(orig)
        elif _has_token(n, 't'):
            t_candidates.append(orig)
    tcol = t_candidates[-1] if t_candidates else None  # 通常温度在右侧

    # x/y/z 候选（名字匹配；排除 index 等）
    x_name_cands = []
    y_name_cands = []
    z_name_cands = []
    for orig, n in zip(raw, low):
        if 'index' in n:   # 避免把 index 误判为 x
            continue
        if _has_token(n, 'x') or any(k in n for k in ['xlocation', 'xcoord', 'xposition', 'posx', 'xpos', '坐标x', 'x坐标']):
            x_name_cands.append(orig)
        if _has_token(n, 'y') or any(k in n for k in ['ylocation', 'ycoord', 'yposition', 'posy', 'ypos', '坐标y', 'y坐标']):
            y_name_cands.append(orig)
        if _has_token(n, 'z') or any(k in n for k in ['zlocation', 'zcoord', 'zposition', 'posz', 'zpos', '坐标z', 'z坐标']):
            z_name_cands.append(orig)

    xcol = x_name_cands[-1] if x_name_cands else None
    ycol = y_name_cands[-1] if y_name_cands else None
    zcol = z_name_cands[-1] if z_name_cands else None

    # 2) 若温度列仍未识别，尝试从数值列启发式选择
    df_num = _to_numeric(df)
    numeric_cols = [c for c in df_num.columns if pd.api.types.is_numeric_dtype(df_num[c])]

    # 剔除明显索引列
    numeric_cols = [c for c in numeric_cols if not _is_index_like(df_num[c])]

    if tcol is None:
        name_hit = [c for c in numeric_cols if any(k in _norm(c) for k in ['temp', 'temperature', '温度']) or _has_token(_norm(c), 't')]
        if name_hit:
            tcol = name_hit[-1]
        else:
            if len(numeric_cols) == 0:
                raise ValueError("未找到温度列；请确保快照包含温度列（如 T / Temperature / 温度 / Temperature_C）")
            tcol = numeric_cols[-1]

    # 3) 若坐标列没找到，从数值列里挑三个“最像坐标”的列（高方差、互相关小）
    coord_cols = [c for c in numeric_cols if c != tcol]
    if xcol is None or ycol is None or zcol is None:
        if len(coord_cols) < 3:
            raise ValueError("无法从表头或数值列中推断出 x/y/z，请检查快照是否包含坐标三列。")
        # 先按方差从大到小挑前 5 个
        variances = [(c, float(pd.Series(df_num[c]).var(skipna=True))) for c in coord_cols]
        variances.sort(key=lambda x: x[1], reverse=True)
        top = [c for c, _ in variances[:5]]

        # 简单启发：取前三个作为 x/y/z（避免和温度高度相关）
        chosen = top[:3] if len(top) >= 3 else (coord_cols[:3])
        if xcol is None: xcol = chosen[0]
        if ycol is None: ycol = chosen[1]
        if zcol is None: zcol = chosen[2]

    return xcol, ycol, zcol, tcol, rcol

# ============ 读取快照 & 构POD ============

def _read_snapshot(path: str):
    # 关闭低内存模式，避免 dtype 警告；混合类型后续我们再强制转数值
    df = _read_csv(path)
    xcol, ycol, zcol, tcol, rcol = _pick_cols(df)

    # 强制数值化
    x = pd.to_numeric(df[xcol], errors='coerce').values
    y = pd.to_numeric(df[ycol], errors='coerce').values
    z = pd.to_numeric(df[zcol], errors='coerce').values
    T = pd.to_numeric(df[tcol], errors='coerce').values

    # 处理 NaN（用均值填充；极端情况下用0）
    def _fill_nan(a):
        s = pd.Series(a)
        if s.isna().all():
            return np.zeros_like(a, dtype=np.float32)
        s = s.fillna(s.mean())
        return s.values.astype(np.float32)

    x = _fill_nan(x); y = _fill_nan(y); z = _fill_nan(z); T = _fill_nan(T)

    regions = df[rcol].astype(str).tolist() if rcol is not None else None
    return x, y, z, T, regions, (xcol, ycol, zcol, tcol, rcol)

def _collect_snapshot_files(
    snapshot_dir: str,
    meta_csv: str | None = None,
    frame_stride: int = 1,
    max_files_per_rpm: int | None = None,
) -> list[str]:
    """
    优先：若存在指定 meta.csv，则按 file 列收集快照；
    否则：扫描目录下 *.csv，自动排除 meta.csv。
    支持 meta 中的相对路径（允许子目录）。
    """
    if meta_csv is None:
        meta_path = os.path.join(snapshot_dir, "meta.csv")
    else:
        meta_path = meta_csv if os.path.isabs(meta_csv) else os.path.join(snapshot_dir, meta_csv)

    files: list[str] = []
    if os.path.exists(meta_path):
        m = _read_csv(meta_path)
        if "file" not in m.columns:
            raise ValueError(f"{meta_path} 缺少 file 列")
        # 若包含 rpm/time_index，则按每个 rpm 进行帧采样与限量
        if {"rpm", "time_index"}.issubset(m.columns):
            files_grouped: list[str] = []
            for rpm, g in m.groupby("rpm"):
                g_sorted = g.sort_values("time_index")
                g_sampled = g_sorted.iloc[::max(1, int(frame_stride))]
                if max_files_per_rpm is not None:
                    g_sampled = g_sampled.iloc[:max_files_per_rpm]
                for f in g_sampled["file"].astype(str).tolist():
                    p = f if os.path.isabs(f) else os.path.join(snapshot_dir, f)
                    if os.path.isfile(p) and p.lower().endswith(".csv"):
                        files_grouped.append(p)
                    else:
                        print(f"meta 文件指向的路径无效或非 CSV：{p}（已跳过）")
            files = files_grouped
        else:
            for f in m["file"].astype(str).tolist():
                # meta 中允许相对路径（含子目录）
                p = f if os.path.isabs(f) else os.path.join(snapshot_dir, f)
                if os.path.isfile(p) and p.lower().endswith(".csv"):
                    files.append(p)
                else:
                    print(f"meta 文件指向的路径无效或非 CSV：{p}（已跳过）")
    else:
        for fn in sorted(os.listdir(snapshot_dir)):
            if not fn.lower().endswith(".csv"):
                continue
            if fn.lower() == "meta.csv":
                continue
            files.append(os.path.join(snapshot_dir, fn))
    return files


def build_pod_from_folder(
    snapshot_dir: str,
    save_dir: str,
    r: int | None = None,
    energy_keep: float = 0.999,
    meta_csv: str | None = None,
    frame_stride: int = 1,
    max_files_per_rpm: int | None = None,
    progress_every: int = 100,
    dtype: str = "float32",
):
    """
    从 snapshot_dir 读取所有快照（同一网格、同节点顺序），
    组装温度矩阵进行 POD（SVD），保存:
      - save_dir/pod_phi.npy  (N, r)
      - save_dir/pod_mean.npy (N,)
      - save_dir/pod_summary.csv  快照列表与能量占比信息、列选择记录
    参数:
      r: 指定模态数；若为 None 则按能量占比 energy_keep 自动截断
      energy_keep: 能量保持率（0~1）
      meta_csv: 指定用于快照清单的 meta 文件（可含子目录路径）
      frame_stride: 帧采样步长（降低读取与内存）
      max_files_per_rpm: 每个工况最多使用的帧数
      progress_every: 读取进度打印频率
      dtype: 内部计算与存储精度，默认 float32
    """
    files = _collect_snapshot_files(
        snapshot_dir,
        meta_csv=meta_csv,
        frame_stride=frame_stride,
        max_files_per_rpm=max_files_per_rpm,
    )
    if len(files) == 0:
        raise FileNotFoundError(f"{snapshot_dir} 下未找到任何快照 CSV（请检查 meta.csv 或 case_*.csv）")

    print(f"将使用 {len(files)} 个快照（stride={frame_stride}, max_per_rpm={max_files_per_rpm}）进行 POD …")

    # 第一遍：计算均值和基本信息（避免一次性加载所有数据）
    print("第一遍：计算快照均值和统计信息...")
    x0, y0, z0, T0, regions0, cols0 = _read_snapshot(files[0])
    N = T0.shape[0]
    M = len(files)
    dtype_np = np.float32 if dtype == "float32" else np.float64

    # 使用增量方式计算均值
    T_sum = np.zeros(N, dtype=dtype_np)
    names = []
    cols_used = []

    for j, f in enumerate(files):
        _, _, _, T, regions, cols = _read_snapshot(f)
        if T.shape[0] != N:
            raise ValueError(f"节点数不一致: {os.path.basename(files[0])}={N}, {os.path.basename(f)}={T.shape[0]}")

        T_sum += T.astype(dtype_np)
        names.append(os.path.relpath(f, snapshot_dir))
        cols_used.append((os.path.basename(f), cols))
        if progress_every and (j % progress_every == 0):
            print(f"  已处理 {j+1}/{M} 计算均值...")

    Tbar_vec = T_sum / M  # [N]
    Tbar = Tbar_vec.reshape(-1, 1)  # [N,1]

    # 第二遍：计算POD基（分批处理避免内存溢出）
    print("第二遍：计算POD基（分批处理大规模数据）...")

    # 自动估计 r（基于能量保持率）
    if r is None:
        # 使用采样方法估计能量分布（避免一次性处理所有数据）
        n_est = min(M, 200)  # 增加到200以更准确估计
        try:
            # 采样计算能量估计
            sample_indices = np.random.choice(M, size=min(M, 1000), replace=False)
            sample_snapshots = []

            for idx in sample_indices:
                f = files[idx]
                _, _, _, T, _, _ = _read_snapshot(f)
                A_sample = (T.astype(dtype_np) - Tbar_vec).reshape(-1, 1)
                sample_snapshots.append(A_sample)

            # 合并采样数据用于能量估计
            A_sample = np.concatenate(sample_snapshots, axis=1)  # [N, sample_size]

            print(f"  正在估计能量分布（基于 {len(sample_indices)} 个采样快照）...")
            _, S_est, _ = randomized_svd(A_sample, n_components=min(n_est, len(sample_indices)), random_state=42)
            energy_est = S_est ** 2

            # 估算总能量：用 Frobenius 范数
            total_energy_est = np.linalg.norm(A_sample, 'fro') ** 2
            cum_energy = np.cumsum(energy_est)
            cum_ratio = cum_energy / total_energy_est

            # 找到第一个超过 energy_keep 的位置
            r = int(np.searchsorted(cum_ratio, energy_keep) + 1)
            r = max(5, min(r, n_est))  # 至少5个模态，避免r=1

            actual_ratio = cum_ratio[r-1] if r <= len(cum_ratio) else cum_ratio[-1]
            print(f"  估计需要 r={r} 个模态以保持 {actual_ratio*100:.2f}% 能量（目标 {energy_keep*100:.1f}%）")
        except Exception as e:
            print(f"  自动估计 r 失败：{e}，使用 r=min(50, M)")
            r = min(50, M)

    # 使用采样Randomized SVD（避免内存溢出，同时获得更好的POD基）
    print(f"  执行采样 Randomized SVD (r={r})...")

    # 随机采样快照用于构建POD基（采样率约10-20%）
    sample_ratio = min(0.15, 1000 / M)  # 最多采样1000个快照
    sample_size = max(500, int(M * sample_ratio))
    sample_size = min(sample_size, M)

    print(f"  采样 {sample_size}/{M} 个快照构建POD基...")

    # 随机选择采样索引
    sample_indices = np.random.choice(M, size=sample_size, replace=False)
    sample_indices.sort()  # 排序便于处理

    # 读取采样快照
    sample_snapshots = []
    for idx in sample_indices:
        f = files[idx]
        _, _, _, T, _, _ = _read_snapshot(f)
        # 去均值并转换为列向量
        A_col = (T.astype(dtype_np) - Tbar_vec).reshape(-1, 1)
        sample_snapshots.append(A_col)

    # 合并采样数据
    A_sample = np.concatenate(sample_snapshots, axis=1)  # [N, sample_size]

    # 对采样数据计算Randomized SVD获得POD基
    print(f"  计算采样数据的Randomized SVD...")
    U_sample, S_sample, Vt_sample = randomized_svd(
        A_sample, n_components=r, n_iter=5, random_state=42
    )

    Phi = U_sample.copy()  # [N, r]
    S_full = S_sample.copy()

    # 现在计算所有快照的POD系数（分批处理）
    batch_size = min(1000, M)  # 每批处理1000个快照（增大批次减少批次数）
    all_coeffs = []

    for i in range(0, M, batch_size):
        end_idx = min(i + batch_size, M)
        print(f"  计算系数批次 {i//batch_size + 1}/{(M + batch_size - 1)//batch_size} ({end_idx - i} 个快照)...")

        # 读取当前批次的数据
        batch_snapshots = []
        for j in range(i, end_idx):
            f = files[j]
            _, _, _, T, _, _ = _read_snapshot(f)
            # 去均值并转换为列向量
            A_col = (T.astype(dtype_np) - Tbar_vec).reshape(-1, 1)
            batch_snapshots.append(A_col)

        # 合并批次数据
        A_batch = np.concatenate(batch_snapshots, axis=1)  # [N, batch_size]

        # 投影到POD基获得系数：coeffs = Phi^T @ A_batch
        coeffs_batch = Phi.T @ A_batch  # [r, N] @ [N, batch_size] = [r, batch_size]
        coeffs_batch = coeffs_batch.T  # [batch_size, r]
        all_coeffs.append(coeffs_batch)

    # 合并所有批次的系数
    coeffs = np.concatenate(all_coeffs, axis=0)  # [M, r]

    # 保存POD结果
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "pod_phi.npy"), Phi)
    np.save(os.path.join(save_dir, "pod_mean.npy"), Tbar_vec)
    np.save(os.path.join(save_dir, "pod_coeffs.npy"), coeffs.astype(dtype_np, copy=False))

    # 额外保存：坐标（来自第一帧），方便物理残差使用
    coords = np.stack([x0.astype(dtype_np), y0.astype(dtype_np), z0.astype(dtype_np)], axis=0)  # [3,N]
    np.save(os.path.join(save_dir, "pod_coords.npy"), coords)

    # 保存摘要（含：能量、每个文件识别到的列名）
    energy = (S_full**2)
    cum = np.cumsum(energy) / (np.sum(energy) + 1e-12)
    keep = cum[r-1] if r-1 < len(cum) else 1.0
    summary = pd.DataFrame({"file": names})
    summary["selected_columns"] = [str(c) for _, c in cols_used]
    summary.attrs['r'] = r
    summary.attrs['energy_keep'] = float(keep)
    summary.to_csv(os.path.join(save_dir, "pod_summary.csv"), index=False, encoding="utf-8-sig")

    print(f"POD 完成：N={N}, M={len(files)}, r={r}, 能量保持≈{keep*100:.2f}%")
    print(f"   已保存：{os.path.join(save_dir,'pod_phi.npy')} ，{os.path.join(save_dir,'pod_mean.npy')}")
    print(f"   已保存：{os.path.join(save_dir,'pod_coeffs.npy')} ，{os.path.join(save_dir,'pod_coords.npy')}")
    print(f"   已记录每个快照选择到的列名 → {os.path.join(save_dir,'pod_summary.csv')}")

    return {
        "phi_path": os.path.join(save_dir, "pod_phi.npy"),
        "mean_path": os.path.join(save_dir, "pod_mean.npy"),
        "coeffs_path": os.path.join(save_dir, "pod_coeffs.npy"),
        "coords_path": os.path.join(save_dir, "pod_coords.npy"),
        "summary_path": os.path.join(save_dir, "pod_summary.csv"),
        "r": r,
        "energy_keep": float(keep),
        "num_snapshots": len(files),
    }
