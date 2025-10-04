import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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


def _normalize_name(name: str) -> str:
    s = str(name).strip().lower()
    replacements = {
        '（': '(', '）': ')', '[': '(', ']' : ')', '℃': 'c', '°': '', '：': ':', '，': ',', '。': '.', '、': ',',
    }
    for src, tgt in replacements.items():
        s = s.replace(src, tgt)
    for ch in [' ', '\u3000', '_', '-', '/', '\\', '.', ',', ':', ';', '%']:
        s = s.replace(ch, '')
    return s


def _safe_pick(df: pd.DataFrame, candidates: List[str]) -> str:
    cols = list(df.columns)
    for name in candidates:
        if name in cols:
            return name

    norm_cols = [_normalize_name(c) for c in cols]
    norm_candidates = [_normalize_name(c) for c in candidates]

    for cand_norm in norm_candidates:
        for col, norm_col in zip(cols, norm_cols):
            if not cand_norm:
                continue
            if norm_col == cand_norm or norm_col.startswith(cand_norm) or cand_norm in norm_col:
                return col
    raise KeyError(f"缺少列：{candidates}，现有列：{list(df.columns)}")


def _unit_scale(colname: str) -> float:
    name = str(colname).lower()
    if "(mm)" in name or name.endswith("_mm"):
        return 1e-3
    return 1.0


def load_phi_mean(phi_path: str, mean_path: str) -> Tuple[np.ndarray, np.ndarray]:
    Phi = np.load(phi_path)
    Tbar = np.load(mean_path)
    return Phi.astype(np.float32), Tbar.astype(np.float32)


def compute_pod_coeff(T: np.ndarray, Phi: np.ndarray, Tbar: np.ndarray) -> np.ndarray:
    return Phi.T @ (T - Tbar)


class SnapshotSequenceDataset(Dataset):
    """
    将瞬态快照序列切片为长度 L 的输入和长度 H 的预测目标。
    支持读取 sensors_map.csv 与 data/real_steady/{rpm}.csv 作为传感器监督。
    """

    def __init__(
        self,
        snapshots_dir: str,
        meta_csv: str,
        phi_path: str,
        mean_path: str,
        sequence_length: int = 20,
        horizon: int = 1,
        stride: int = 1,
        condition_cols: Optional[List[str]] = None,
        sensors_map_csv: Optional[str] = None,
        real_steady_dir: Optional[str] = None,
    ):
        super().__init__()
        self.snapshots_dir = snapshots_dir
        self.meta = _read_csv(meta_csv)
        if not {"file", "rpm", "ambient", "time_index", "time_seconds"}.issubset(self.meta.columns):
            raise ValueError("meta_transient.csv 需要列: file, rpm, ambient, time_index, time_seconds")
        self.meta.sort_values(["rpm", "time_index"], inplace=True)
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.stride = stride

        self.condition_cols = condition_cols or ["rpm", "ambient"]
        missing = [c for c in self.condition_cols if c not in self.meta.columns]
        if missing:
            raise KeyError(f"meta_transient.csv 缺少条件列: {missing}")

        # 传感器配置
        self.sensors_map_csv = sensors_map_csv
        self.real_steady_dir = real_steady_dir
        self.sensor_names: List[str] = []
        self.sensor_count: int = 0
        if self.sensors_map_csv is not None and self.real_steady_dir is not None:
            smap = _read_csv(self.sensors_map_csv)
            req = {"name", "x", "y", "z"}
            if not req.issubset(smap.columns):
                raise RuntimeError(f"sensors_map.csv 需要列：{req}")
            self.sensor_names = smap["name"].astype(str).tolist()
            self.sensor_count = len(self.sensor_names)

        self.Phi, self.Tbar = load_phi_mean(phi_path, mean_path)
        self.r = self.Phi.shape[1]

        self.coords: Optional[np.ndarray] = None  # [3, N]

        # 预读取全部快照的 POD 系数 + 传感器序列
        self.data_by_rpm: Dict[int, Dict[str, np.ndarray]] = {}
        self._preload_all()
        self.windows = self._build_windows()

    def _read_snapshot(self, fpath: str):
        df = _read_csv(fpath)
        tcol = _safe_pick(df, ["Temperature", "temperature", "T", "Temp", "温度", "Temperature_C"])
        xcol = _safe_pick(df, ["x", "X", "x_mm", "X(mm)"])
        ycol = _safe_pick(df, ["y", "Y", "y_mm", "Y(mm)"])
        zcol = _safe_pick(df, ["z", "Z", "z_mm", "Z(mm)"])

        x = df[xcol].to_numpy(np.float32) * _unit_scale(xcol)
        y = df[ycol].to_numpy(np.float32) * _unit_scale(ycol)
        z = df[zcol].to_numpy(np.float32) * _unit_scale(zcol)
        T = df[tcol].to_numpy(np.float32)
        if np.isnan(T).any():
            T = np.nan_to_num(T, nan=np.nanmean(T))
        return x, y, z, T

    def _ensure_coords(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        if self.coords is None:
            self.coords = np.stack([x, y, z], axis=0)
        else:
            if not (len(x) == self.coords.shape[1] == len(y) == len(z)):
                raise ValueError("不同快照的节点数量不一致")

    def _load_real_steady_series(self, rpm: int) -> Optional[np.ndarray]:
        if self.real_steady_dir is None or self.sensor_count == 0:
            return None
        path = os.path.join(self.real_steady_dir, f"{int(rpm)}.csv")
        if not os.path.exists(path):
            return None
        df = _read_csv(path)
        # 找到列：温度值1..温度值S（按数字序排列）
        cols = []
        for c in df.columns:
            m = re.match(r"\s*温度值\s*(\d+)\s*", str(c))
            if m:
                idx = int(m.group(1))
                cols.append((idx, c))
        if not cols:
            # 兼容：英文或其他命名
            tmp = [(i + 1, c) for i, c in enumerate(df.columns) if _normalize_name(c).startswith("温度")]
            cols = tmp
        cols.sort(key=lambda x: x[0])
        cols = [c for _, c in cols][: self.sensor_count]
        if not cols:
            return None
        arr = df[cols].to_numpy(np.float32)
        return arr  # [T, S]

    def _preload_all(self):
        for rpm, group in self.meta.groupby("rpm"):
            coeff_list = []
            cond_list = []
            times = []

            sensor_seq = self._load_real_steady_series(int(rpm))

            for i, (_, row) in enumerate(group.iterrows()):
                fpath = os.path.join(self.snapshots_dir, row["file"])
                x, y, z, T = self._read_snapshot(fpath)
                self._ensure_coords(x, y, z)

                a = compute_pod_coeff(T, self.Phi, self.Tbar)
                coeff_list.append(a.astype(np.float32))

                cond = row[self.condition_cols].to_numpy(np.float32)
                cond_list.append(cond)

                times.append((int(row["time_index"]), float(row["time_seconds"])))

            coeff_arr = np.stack(coeff_list)  # [F, r]
            cond_arr = np.stack(cond_list)    # [F, d]

            # 传感器序列长度与帧数对齐（截断到最短）
            if sensor_seq is not None:
                min_len = min(coeff_arr.shape[0], sensor_seq.shape[0])
                coeff_arr = coeff_arr[:min_len]
                cond_arr = cond_arr[:min_len]
                times = times[:min_len]
                sensor_seq = sensor_seq[:min_len]

            self.data_by_rpm[int(rpm)] = {
                "coeff": coeff_arr,
                "cond": cond_arr,
                "times": times,
                "sensors": sensor_seq,  # 可能为 None
            }

    def _build_windows(self):
        windows = []
        for rpm, payload in self.data_by_rpm.items():
            coeff = payload["coeff"]
            num_frames = coeff.shape[0]
            max_start = num_frames - (self.sequence_length + self.horizon)
            if max_start < 0:
                continue
            for start in range(0, max_start + 1, self.stride):
                end = start + self.sequence_length
                target_end = end + self.horizon
                windows.append((rpm, start, end, target_end))
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rpm, start, end, target_end = self.windows[idx]
        payload = self.data_by_rpm[rpm]
        coeff = payload["coeff"][start:end]
        target = payload["coeff"][end:target_end]
        cond = payload["cond"][end - 1]
        time_index, time_seconds = payload["times"][end - 1]

        item = {
            "seq_a": torch.from_numpy(coeff),  # [L, r]
            "target_a": torch.from_numpy(target),  # [H, r]
            "cond": torch.from_numpy(cond),
            "time_index": torch.tensor(time_index, dtype=torch.long),
            "time_seconds": torch.tensor(time_seconds, dtype=torch.float32),
        }
        if payload["sensors"] is not None:
            sensors = payload["sensors"][end:target_end]  # [H, S]
            item["sensor_target"] = torch.from_numpy(sensors)
        return item


class PrecomputedCoeffDataset(Dataset):
    """
    直接加载 POD 预计算产物：pod_coeffs.npy（[M,r]）、pod_mean/phi、pod_coords.npy（[3,N]），
    再结合 meta_transient.csv 构造 (seq -> horizon) 的训练样本，避免重复读 CSV。
    可选传感器监督依然通过 real_steady/{rpm}.csv 读取。
    """
    def __init__(
        self,
        meta_csv: str,
        coeffs_path: str,
        phi_path: str,
        mean_path: str,
        coords_path: str,
        sequence_length: int = 60,
        horizon: int = 1,
        stride: int = 1,
        condition_cols: Optional[List[str]] = None,
        sensors_map_csv: Optional[str] = None,
        real_steady_dir: Optional[str] = None,
    ):
        super().__init__()
        self.meta = _read_csv(meta_csv)
        if not {"file", "rpm", "ambient", "time_index", "time_seconds"}.issubset(self.meta.columns):
            raise ValueError("meta_transient.csv 需要列: file, rpm, ambient, time_index, time_seconds")
        self.meta.sort_values(["rpm", "time_index"], inplace=True)

        self.Phi, self.Tbar = load_phi_mean(phi_path, mean_path)
        self.coords = np.load(coords_path).astype(np.float32)  # [3,N]
        self.coeffs_all = np.load(coeffs_path).astype(np.float32)  # [M, r]
        self.r = self.Phi.shape[1]

        self.sequence_length = sequence_length
        self.horizon = horizon
        self.stride = stride
        self.condition_cols = condition_cols or ["rpm", "ambient"]

        # 传感器映射
        self.sensors_map_csv = sensors_map_csv
        self.real_steady_dir = real_steady_dir
        self.sensor_names: List[str] = []
        if self.sensors_map_csv is not None:
            smap = _read_csv(self.sensors_map_csv)
            if not {"name","x","y","z"}.issubset(smap.columns):
                raise RuntimeError("sensors_map.csv 需要列: name,x,y,z")
            self.sensor_names = smap["name"].astype(str).tolist()

        # 将 coeffs 切分到每个 rpm 序列（按 meta 顺序）
        self.data_by_rpm: Dict[int, Dict[str, np.ndarray]] = {}
        self._split_by_rpm()
        self.windows = self._build_windows()

    def _load_real_steady_series(self, rpm: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if self.real_steady_dir is None or not self.sensor_names:
            return None
        path = os.path.join(self.real_steady_dir, f"{int(rpm)}.csv")
        if not os.path.exists(path):
            return None
        df = _read_csv(path)
        # 读取时间列（秒）
        t_rs = None
        for c in ["time_seconds", "second", "seconds", "t", "时间(s)", "时间", "timestamp", "Timestamp"]:
            if c in df.columns:
                try:
                    if c in ["时间", "timestamp", "Timestamp"]:
                        ts = pd.to_datetime(df[c], errors='coerce')
                        base = ts.iloc[0]
                        t_rs = (ts - base).dt.total_seconds().to_numpy(np.float32)
                    else:
                        t_rs = pd.to_numeric(df[c], errors='coerce').to_numpy(np.float32)
                except Exception:
                    t_rs = None
                break
        if t_rs is None:
            t_rs = np.arange(len(df), dtype=np.float32)
        
        # 按索引顺序读取传感器列：温度值1, 温度值2, ..., 温度值N
        # sensors_map.csv 的行顺序对应温度值的序号
        S = len(self.sensor_names)
        arrs = []
        for i in range(1, S + 1):
            col_name = f"温度值{i}"
            if col_name in df.columns:
                arrs.append(pd.to_numeric(df[col_name], errors='coerce').to_numpy(np.float32))
            else:
                # 尝试英文命名：Temp1, Temperature1, Sensor1 等
                for alt in [f"Temp{i}", f"Temperature{i}", f"Sensor{i}", f"T{i}"]:
                    if alt in df.columns:
                        arrs.append(pd.to_numeric(df[alt], errors='coerce').to_numpy(np.float32))
                        break
                else:
                    # 找不到则填 NaN
                    arrs.append(np.full((len(df),), np.nan, dtype=np.float32))
        
        if not arrs:
            return None
        Traw = min(a.shape[0] for a in arrs)
        A = np.stack([a[:Traw] for a in arrs], axis=1)  # [Traw,S]
        t_rs = t_rs[:Traw]
        return t_rs, A

    def _split_by_rpm(self):
        idx_start = 0
        for rpm, g in self.meta.groupby("rpm"):
            length = len(g)
            coeff = self.coeffs_all[idx_start: idx_start + length]
            cond = g[[c for c in self.condition_cols]].to_numpy(np.float32)
            times = list(zip(g["time_index"].astype(int).tolist(), g["time_seconds"].astype(float).tolist()))
            t_meta = g["time_seconds"].to_numpy(np.float32)

            sensors = None
            rs = self._load_real_steady_series(int(rpm))
            if rs is not None:
                t_rs, A_rs = rs  # [T_rs], [T_rs,S]
                # 最近邻对齐到 meta 的秒
                idx_nn = np.searchsorted(t_rs, t_meta, side='left')
                idx_nn = np.clip(idx_nn, 0, len(t_rs)-1)
                # 比较左右邻更近的一个
                idx_nn = np.where(
                    (idx_nn > 0) & ((np.abs(t_rs[idx_nn] - t_meta) >= np.abs(t_rs[idx_nn-1] - t_meta))),
                    idx_nn - 1,
                    idx_nn,
                )
                sensors = A_rs[idx_nn]
                # 统一长度
                min_len = min(length, sensors.shape[0])
                coeff = coeff[:min_len]
                cond = cond[:min_len]
                times = times[:min_len]
                sensors = sensors[:min_len]
            else:
                # 无实测则保持 None
                pass

            self.data_by_rpm[int(rpm)] = {
                "coeff": coeff,
                "cond": cond,
                "times": times,
                "sensors": sensors,  # [T,S] or None
            }
            idx_start += length

    def _build_windows(self):
        windows = []
        for rpm, payload in self.data_by_rpm.items():
            coeff = payload["coeff"]
            num_frames = coeff.shape[0]
            max_start = num_frames - (self.sequence_length + self.horizon)
            if max_start < 0:
                continue
            for start in range(0, max_start + 1, self.stride):
                end = start + self.sequence_length
                target_end = end + self.horizon
                windows.append((rpm, start, end, target_end))
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rpm, start, end, target_end = self.windows[idx]
        payload = self.data_by_rpm[rpm]
        coeff = payload["coeff"][start:end]
        target = payload["coeff"][end:target_end]
        cond = payload["cond"][end - 1]
        time_index, time_seconds = payload["times"][end - 1]

        item = {
            "seq_a": torch.from_numpy(coeff),
            "target_a": torch.from_numpy(target),
            "cond": torch.from_numpy(cond),
            "time_index": torch.tensor(time_index, dtype=torch.long),
            "time_seconds": torch.tensor(time_seconds, dtype=torch.float32),
            "start_time_index": torch.tensor(start, dtype=torch.long),
            "is_t0": torch.tensor(1 if start == 0 else 0, dtype=torch.long),
        }
        if payload["sensors"] is not None:
            sensor_h = payload["sensors"][end:target_end]
            item["sensor_target"] = torch.from_numpy(sensor_h.astype(np.float32))
        return item
