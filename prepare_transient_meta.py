import argparse
import os
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _parse_rpm_from_dir(dirname: str) -> int:
    match = re.search(r"(\d+)", dirname)
    if not match:
        raise ValueError(f"无法从目录名 {dirname} 中解析转速")
    return int(match.group(1))


def build_transient_meta(
    snapshots_dir: Path,
    base_meta_path: Path,
    output_path: Path,
    delta_t: float = 1.0,
    file_pattern: str = "temp_",
) -> pd.DataFrame:
    if not base_meta_path.exists():
        raise FileNotFoundError(f"未找到基础元信息文件：{base_meta_path}")

    base_meta = pd.read_csv(base_meta_path)
    if "rpm" not in base_meta.columns:
        raise ValueError("基础 meta.csv 缺少 'rpm' 列")

    base_meta = base_meta.set_index("rpm")

    rows: List[Dict] = []

    for entry in sorted(snapshots_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.lower() == "__pycache__":
            continue

        # 从目录名解析转速
        try:
            rpm = _parse_rpm_from_dir(entry.name)
        except ValueError:
            continue

        if rpm not in base_meta.index:
            raise KeyError(f"meta.csv 中缺少转速 {rpm} 的记录")

        base_row = base_meta.loc[rpm].to_dict()
        ambient = float(base_row.get("ambient", 0.0))

        csv_files = sorted(
            f for f in entry.iterdir()
            if f.is_file() and f.suffix.lower() == ".csv" and f.name.lower() != "meta.csv"
        )
        csv_files = [f for f in csv_files if f.name.startswith(file_pattern) or file_pattern == ""]
        if not csv_files:
            raise FileNotFoundError(f"目录 {entry} 中未找到匹配 '{file_pattern}*.csv' 的快照文件")

        for idx, fpath in enumerate(csv_files):
            time_seconds = idx * delta_t
            row = {
                "file": fpath.relative_to(snapshots_dir).as_posix(),
                "rpm": rpm,
                "ambient": ambient,
                "time_index": idx,
                "time_seconds": time_seconds,
            }
            # 合并基础 meta 的其余列
            for key, value in base_row.items():
                if key in ("file", "rpm", "ambient"):
                    continue
                row[key] = value
            rows.append(row)

    if not rows:
        raise RuntimeError("未收集到任何快照记录，请检查目录结构和文件命名。")

    df = pd.DataFrame(rows)
    # 排列列顺序：核心列在前
    front_cols = ["file", "rpm", "ambient", "time_index", "time_seconds"]
    other_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + other_cols]

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df


def main():
    parser = argparse.ArgumentParser(description="生成瞬态快照的元信息表 (meta_transient.csv)")
    parser.add_argument("--snapshots-dir", default="data/snapshots", help="快照根目录")
    parser.add_argument("--base-meta", default="data/snapshots/meta.csv", help="基础 meta.csv 路径")
    parser.add_argument("--output", default="data/snapshots/meta_transient.csv", help="输出文件路径")
    parser.add_argument("--delta-t", type=float, default=1.0, help="相邻快照的时间步长 (秒)")
    parser.add_argument("--file-pattern", default="temp_", help="快照文件名前缀 (留空则不筛选)")

    args = parser.parse_args()

    snapshots_dir = Path(args.snapshots_dir).resolve()
    base_meta_path = Path(args.base_meta).resolve()
    output_path = Path(args.output).resolve()

    if not snapshots_dir.exists():
        raise FileNotFoundError(f"快照目录不存在：{snapshots_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_transient_meta(
        snapshots_dir=snapshots_dir,
        base_meta_path=base_meta_path,
        output_path=output_path,
        delta_t=args.delta_t,
        file_pattern=args.file_pattern,
    )

    print(f"已生成 {len(df)} 条记录：{output_path}")


if __name__ == "__main__":
    main()

