import os
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import pyvista as pv  # 仅用于读取 STL 并提取三角面

DEFAULT_DIR = "predict_visualization_results/11000_csv"
DEFAULT_STL = "data/model.stl"
DEFAULT_OUT = "predict_visualization_results/sequence_viewer_2.html"
DEFAULT_POINT_STRIDE = 5     # 点抽样（每N个点取1个）
DEFAULT_FRAME_STRIDE = 1    # 帧抽样（每N帧取1帧）
DEFAULT_MAX_FRAMES = None     # 限制最大帧数，避免HTML过大
DEFAULT_FPS = 10             # 播放帧率（帧/秒）


def _read_csv(path: str, nrows: int | None = None) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gbk", "ansi"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, nrows=nrows)
        except Exception:
            continue
    return pd.read_csv(path, low_memory=False, nrows=nrows)


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    low = [str(c).strip().lower().replace(' ', '') for c in cols]
    cand_low = [s.strip().lower().replace(' ', '') for s in candidates]
    for i, name in enumerate(low):
        for cand in cand_low:
            if cand and cand in name:
                return cols[i]
    raise KeyError(f"列不存在（任取其一）：{candidates}；现有列：{cols}")


def load_meta(seq_dir: str) -> pd.DataFrame:
    meta_path = os.path.join(seq_dir, "meta_pred.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"未找到 {meta_path}")
    meta = _read_csv(meta_path)
    if "time_index" in meta.columns:
        meta = meta.sort_values("time_index")
    return meta.reset_index(drop=True)


def compute_global_minmax(files: list[str], seq_dir: str, step: int) -> tuple[float, float]:
    vmin, vmax = np.inf, -np.inf
    for i in range(0, len(files), max(1, int(step))):
        f = os.path.join(seq_dir, files[i])
        df = _read_csv(f)
        tcol = _pick_col(df, ["Temperature (°C)", "Temperature", "temperature", "温度", "Temp"])
        arr = pd.to_numeric(df[tcol], errors='coerce').to_numpy(np.float32)
        if arr.size:
            vmin = min(vmin, float(np.nanmin(arr)))
            vmax = max(vmax, float(np.nanmax(arr)))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    return vmin, vmax


def load_stl_mesh(stl_path: str):
    if not os.path.exists(stl_path):
        return None
    try:
        surf = pv.read(stl_path)
        pts = np.array(surf.points)
        faces = surf.faces.reshape(-1, 4)[:, 1:4]  # (n,3)
        return pts, faces
    except Exception:
        return None


def build_html(seq_dir: str, stl_path: str, out_html: str,
               point_stride: int, frame_stride: int, max_frames: int, fps: int):
    meta = load_meta(seq_dir)
    files_all = meta["file"].astype(str).tolist()

    # 为每个文件建立秒数映射（优先 time_seconds，否则 time_index，否则按顺序编号）
    if "time_seconds" in meta.columns:
        sec_map = {str(r["file"]): float(r["time_seconds"]) for _, r in meta.iterrows()}
    elif "time_index" in meta.columns:
        sec_map = {str(r["file"]): int(r["time_index"]) for _, r in meta.iterrows()}
    else:
        sec_map = {f: i for i, f in enumerate(files_all)}

    # 帧抽样
    files = files_all[::max(1, int(frame_stride))]
    if max_frames is not None:
        files = files[:int(max_frames)]

    # 取第一帧几何 & 列名
    f0 = os.path.join(seq_dir, files[0])
    df0 = _read_csv(f0)
    xcol = _pick_col(df0, ["X Location (m)", "x", "X", "x_mm", "X(mm)"])
    ycol = _pick_col(df0, ["Y Location (m)", "y", "Y", "y_mm", "Y(mm)"])
    zcol = _pick_col(df0, ["Z Location (m)", "z", "Z", "z_mm", "Z(mm)"])
    tcol = _pick_col(df0, ["Temperature (°C)", "Temperature", "temperature", "温度", "Temp"])

    x = pd.to_numeric(df0[xcol], errors='coerce').to_numpy(np.float32)
    y = pd.to_numeric(df0[ycol], errors='coerce').to_numpy(np.float32)
    z = pd.to_numeric(df0[zcol], errors='coerce').to_numpy(np.float32)

    # 点抽样索引
    N = x.shape[0]
    step_points = max(1, int(point_stride))
    idx_keep = np.arange(0, N, step_points, dtype=np.int64)

    # 全局色域
    vmin, vmax = compute_global_minmax(files, seq_dir, step=max(1, int(len(files)/10)))

    # 初始温度
    t0 = pd.to_numeric(df0[tcol], errors='coerce').to_numpy(np.float32)

    # STL（可选）
    stl = load_stl_mesh(stl_path)

    # Plotly figure
    fig = go.Figure()

    # STL Mesh（若存在）
    if stl is not None:
        pts, faces = stl
        fig.add_trace(go.Mesh3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color='lightgray', opacity=0.2, name='surface', showscale=False
        ))

    # 初始散点
    fig.add_trace(go.Scatter3d(
        x=x[idx_keep], y=y[idx_keep], z=z[idx_keep],
        mode='markers',
        marker=dict(
            size=2,
            color=t0[idx_keep],
            colorscale='Plasma',
            cmin=vmin, cmax=vmax,
            colorbar=dict(title='Temperature (°C)')
        ),
        name='temperature'
    ))

    # 记录温度散点的 trace 索引（若前面加入了 STL，则此索引为 1，否则为 0）
    scatter_idx = len(fig.data) - 1

    # 构造 frames（使用真实秒数作为名称与标签）
    frames = []
    steps = []
    for ti, fname in enumerate(files):
        fpath = os.path.join(seq_dir, fname)
        df = _read_csv(fpath)
        vals = pd.to_numeric(df[tcol], errors='coerce').to_numpy(np.float32)[idx_keep]
        tsec = sec_map.get(fname, ti)
        frames.append(go.Frame(
            data=[go.Scatter3d(marker=dict(color=vals, colorscale='Plasma', cmin=vmin, cmax=vmax))],
            traces=[scatter_idx],
            name=str(tsec)
        ))
        steps.append(dict(method='animate', args=[[str(tsec)],
                      dict(frame=dict(duration=0, redraw=True),
                           mode='immediate', transition=dict(duration=0))],
                      label=str(int(tsec))))

    fig.frames = frames

    # 播放时长（ms/帧）
    frame_ms = int(1000 / max(1, int(fps)))

    # 布局与滑块/播放
    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
            aspectmode='data'  # 按实际数据比例显示，避免变形
        ),
        updatemenus=[{
            'type': 'buttons',
            'buttons': [
                {'label': 'Play', 'method': 'animate',
                 'args': [None, {'frame': {'duration': frame_ms, 'redraw': True},
                                  'fromcurrent': True, 'transition': {'duration': 0}}]},
                {'label': 'Pause', 'method': 'animate',
                 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
            ],
            'showactive': False,
            'x': 0.1, 'y': 0
        }],
        sliders=[{
            'steps': steps,
            'x': 0.1, 'len': 0.8,
            'currentvalue': {'prefix': 't = ', 'suffix': ' s', 'visible': True}
        }]
    )

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fig.write_html(out_html, include_plotlyjs='cdn', auto_play=False, full_html=True)
    print(f"已导出 HTML：{out_html}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=DEFAULT_DIR, help='逐秒CSV目录（包含 meta_pred.csv 和 temp_*.csv）')
    ap.add_argument('--stl', default=DEFAULT_STL, help='STL 模型路径')
    ap.add_argument('--out', default=DEFAULT_OUT, help='输出 HTML 路径')
    ap.add_argument('--point-stride', type=int, default=DEFAULT_POINT_STRIDE)
    ap.add_argument('--frame-stride', type=int, default=DEFAULT_FRAME_STRIDE)
    ap.add_argument('--max-frames', type=int, default=DEFAULT_MAX_FRAMES)
    ap.add_argument('--fps', type=int, default=DEFAULT_FPS)
    args = ap.parse_args()

    build_html(seq_dir=args.dir, stl_path=args.stl, out_html=args.out,
               point_stride=args.point_stride, frame_stride=args.frame_stride,
               max_frames=args.max_frames, fps=args.fps)


if __name__ == '__main__':
    main()
