"""
对比传感器位置的预测温度和真实测量温度
验证模型在实际传感器位置的准确性
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from dataclasses import dataclass
from typing import Tuple, List

# 设置matplotlib支持中文显示
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


@dataclass
class CompareConfig:
    rpm: int = 12000  # 可改为 9000/12000/15000/18000/21000
    sensors_map_csv: str = "data/sensors_map.csv"
    real_data_csv: str = ""  # 留空将根据 rpm 自动设置为 data/real_steady/{rpm}.csv
    pred_csv_dir: str = ""   # 留空将根据 rpm 自动设置为 prediction_results/{rpm}_csv
    output_dir: str = "sensor_comparison_results"
    time_samples: int = 10  # 采样多少个时间点进行对比


def read_csv_auto_encoding(path: str) -> pd.DataFrame:
    """自动尝试不同编码读取CSV"""
    for enc in ["utf-8", "utf-8-sig", "gbk", "ansi"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(path, low_memory=False)


def find_nearest_nodes(sensor_coords: np.ndarray, mesh_coords: np.ndarray) -> np.ndarray:
    """
    为每个传感器找到最近的网格节点
    
    Args:
        sensor_coords: [N_sensors, 3] 传感器坐标
        mesh_coords: [N_nodes, 3] 网格节点坐标
        
    Returns:
        indices: [N_sensors] 最近节点的索引
    """
    indices = []
    for sensor_pos in sensor_coords:
        # 计算与所有节点的距离
        distances = np.sqrt(np.sum((mesh_coords - sensor_pos) ** 2, axis=1))
        nearest_idx = np.argmin(distances)
        indices.append(nearest_idx)
        
        # 输出距离信息
        min_dist = distances[nearest_idx]
        print(f"  传感器 ({sensor_pos[0]:.6f}, {sensor_pos[1]:.6f}, {sensor_pos[2]:.6f}) "
              f"-> 节点 {nearest_idx}, 距离: {min_dist*1000:.3f} mm")
    
    return np.array(indices)


def extract_predicted_temps(pred_csv_dir: str, sensor_node_indices: np.ndarray, 
                           time_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    从预测的CSV文件中提取传感器位置的温度
    
    Returns:
        times: [T] 时间索引数组
        temps: [T, N_sensors] 温度数组
    """
    # 扫描目录中的所有 temp_*.csv 文件
    import glob
    csv_files = sorted(glob.glob(os.path.join(pred_csv_dir, "temp_*.csv")))

    if not csv_files:
        raise FileNotFoundError(f"在 {pred_csv_dir} 中找不到 temp_*.csv 文件")

    # 从文件名中提取时间索引
    all_times = []
    for f in csv_files:
        # 从 temp_0000.csv 提取时间
        basename = os.path.basename(f)
        time_idx = int(basename.split('_')[1].split('.')[0])
        all_times.append(time_idx)
    
    all_times = np.array(all_times)
    
    # 采样时间点
    total_times = len(all_times)
    if time_samples >= total_times:
        time_indices = np.arange(total_times)
    else:
        time_indices = np.linspace(0, total_times - 1, time_samples, dtype=int)
    
    times = all_times[time_indices]
    selected_files = [csv_files[i] for i in time_indices]
    
    # 提取每个时间点的温度
    temps_list = []
    for i, (t, csv_path) in enumerate(zip(times, selected_files)):
        df = read_csv_auto_encoding(csv_path)
        
        # 提取温度列（新格式使用 'T'）
        temp_col = None
        for col in ["T", "Temperature (°C)", "Temperature", "temperature"]:
            if col in df.columns:
                temp_col = col
                break
        
        if temp_col is None:
            raise ValueError(f"找不到温度列: {df.columns}")
        
        # 提取传感器位置的温度
        temps = df.iloc[sensor_node_indices][temp_col].values
        temps_list.append(temps)
        
        print(f"  时间 {t:.0f}s: 已提取温度")
    
    temps = np.array(temps_list)  # [T, N_sensors]
    return times, temps


def extract_real_temps(real_csv: str, sensor_names: List[str], 
                      pred_times: np.ndarray) -> np.ndarray:
    """
    从真实测量数据中提取传感器温度
    
    Returns:
        temps: [T, N_sensors] 温度数组
    """
    df = read_csv_auto_encoding(real_csv)
    
    # 解析时间列
    time_col = df.columns[0]
    timestamps = pd.to_datetime(df[time_col])
    base_time = timestamps.iloc[0]
    time_seconds = (timestamps - base_time).dt.total_seconds().values
    
    # 获取传感器温度列（温度值1-7）
    n_sensors = len(sensor_names)
    temp_cols = [f"温度值{i+1}" for i in range(n_sensors)]
    
    # 检查列是否存在
    missing_cols = [col for col in temp_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"真实数据缺少列: {missing_cols}")
    
    temp_data = df[temp_cols].values  # [T_real, N_sensors]
    
    # 对齐到预测的时间点（最近邻插值）
    aligned_temps = []
    for t_pred in pred_times:
        idx = np.argmin(np.abs(time_seconds - t_pred))
        aligned_temps.append(temp_data[idx])
        print(f"  预测时间 {t_pred:.0f}s -> 真实时间 {time_seconds[idx]:.0f}s")
    
    return np.array(aligned_temps)  # [T, N_sensors]


def plot_comparison(sensor_names: List[str], pred_times: np.ndarray,
                   pred_temps: np.ndarray, real_temps: np.ndarray,
                   output_dir: str, rpm: int):
    """绘制对比图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    n_sensors = len(sensor_names)
    
    # 1. 每个传感器的时间序列对比
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, name in enumerate(sensor_names):
        ax = axes[i]
        ax.plot(pred_times, pred_temps[:, i], 'o-', label='预测值', linewidth=2, markersize=6)
        ax.plot(pred_times, real_temps[:, i], 's-', label='真实值', linewidth=2, markersize=6)
        ax.set_xlabel('时间 (s)', fontsize=11)
        ax.set_ylabel('温度 (°C)', fontsize=11)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(n_sensors, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'传感器温度对比 - {rpm} RPM', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sensor_timeseries_{rpm}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存: sensor_timeseries_{rpm}.png")
    
    # 2. 真实值 vs 预测值散点图
    plt.figure(figsize=(10, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_sensors))
    
    for i, (name, color) in enumerate(zip(sensor_names, colors)):
        plt.scatter(real_temps[:, i], pred_temps[:, i], 
                   s=100, alpha=0.7, label=name, color=color)
    
    # 绘制理想线 y=x
    all_temps = np.concatenate([pred_temps.flatten(), real_temps.flatten()])
    min_temp = all_temps.min()
    max_temp = all_temps.max()
    plt.plot([min_temp, max_temp], [min_temp, max_temp], 'r--', linewidth=2, label='理想预测线')
    
    plt.xlabel('真实温度 (°C)', fontsize=14)
    plt.ylabel('预测温度 (°C)', fontsize=14)
    plt.title(f'传感器温度: 真实值 vs 预测值 - {rpm} RPM', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sensor_scatter_{rpm}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存: sensor_scatter_{rpm}.png")
    
    # 3. 误差分布
    errors = pred_temps - real_temps  # [T, N_sensors]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 3.1 每个传感器的误差箱线图
    ax = axes[0]
    bp = ax.boxplot([errors[:, i] for i in range(n_sensors)], 
                     labels=sensor_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_ylabel('预测误差 (°C)', fontsize=12)
    ax.set_title('传感器误差分布', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3.2 误差直方图
    ax = axes[1]
    ax.hist(errors.flatten(), bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='零误差线')
    ax.axvline(x=errors.mean(), color='green', linestyle='--', linewidth=2, 
              label=f'平均误差: {errors.mean():.3f}°C')
    ax.set_xlabel('预测误差 (°C)', fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    ax.set_title('总体误差分布', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sensor_errors_{rpm}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存: sensor_errors_{rpm}.png")


def calculate_metrics(pred_temps: np.ndarray, real_temps: np.ndarray, 
                     sensor_names: List[str]) -> dict:
    """计算评价指标"""
    errors = pred_temps - real_temps
    abs_errors = np.abs(errors)
    
    metrics = {
        "总体指标": {
            "平均绝对误差 (MAE)": float(abs_errors.mean()),
            "均方根误差 (RMSE)": float(np.sqrt((errors ** 2).mean())),
            "最大绝对误差": float(abs_errors.max()),
            "平均偏差 (Bias)": float(errors.mean()),
        },
        "各传感器指标": {}
    }
    
    for i, name in enumerate(sensor_names):
        sensor_errors = errors[:, i]
        sensor_abs_errors = abs_errors[:, i]
        
        metrics["各传感器指标"][name] = {
            "MAE (°C)": float(sensor_abs_errors.mean()),
            "RMSE (°C)": float(np.sqrt((sensor_errors ** 2).mean())),
            "最大误差 (°C)": float(sensor_abs_errors.max()),
            "平均偏差 (°C)": float(sensor_errors.mean()),
            "标准差 (°C)": float(sensor_errors.std()),
        }
    
    return metrics


def main():
    cfg = CompareConfig()

    # 自动补全路径
    if not cfg.pred_csv_dir:
        cfg.pred_csv_dir = os.path.join("prediction_results", f"{cfg.rpm}_csv")
    if not cfg.real_data_csv:
        cfg.real_data_csv = os.path.join("data", "real_steady", f"{cfg.rpm}.csv")
    
    print("=" * 70)
    print(f"传感器位置温度对比验证 - {cfg.rpm} RPM")
    print("=" * 70)
    
    # 1. 读取传感器位置
    print("\n[1] 读取传感器位置...")
    sensors_df = read_csv_auto_encoding(cfg.sensors_map_csv)
    sensor_names = sensors_df['name'].tolist()
    sensor_coords = sensors_df[['x', 'y', 'z']].values
    print(f"  传感器数量: {len(sensor_names)}")
    for name, coord in zip(sensor_names, sensor_coords):
        print(f"    {name}: ({coord[0]:.6f}, {coord[1]:.6f}, {coord[2]:.6f})")
    
    # 2. 读取一个预测文件，获取网格坐标
    print("\n[2] 匹配传感器到网格节点...")
    import glob
    csv_files = sorted(glob.glob(os.path.join(cfg.pred_csv_dir, "temp_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"在 {cfg.pred_csv_dir} 中找不到 temp_*.csv 文件。\n"
                                f"请先运行导出脚本或检查路径是否为 prediction_results/{cfg.rpm}_csv")
    sample_csv = csv_files[0]  # 使用第一个文件
    mesh_df = read_csv_auto_encoding(sample_csv)
    
    # 自动识别坐标列名
    coord_cols = None
    if 'x' in mesh_df.columns and 'y' in mesh_df.columns and 'z' in mesh_df.columns:
        coord_cols = ['x', 'y', 'z']
    elif 'X Location (m)' in mesh_df.columns:
        coord_cols = ['X Location (m)', 'Y Location (m)', 'Z Location (m)']
    else:
        raise ValueError(f"找不到坐标列，可用列: {mesh_df.columns.tolist()}")
    
    mesh_coords = mesh_df[coord_cols].values
    
    # 找到最近的节点
    sensor_node_indices = find_nearest_nodes(sensor_coords, mesh_coords)
    
    # 3. 提取预测温度
    print(f"\n[3] 提取预测温度 (采样 {cfg.time_samples} 个时间点)...")
    pred_times, pred_temps = extract_predicted_temps(
        cfg.pred_csv_dir, sensor_node_indices, cfg.time_samples
    )
    
    # 4. 提取真实温度
    print(f"\n[4] 提取真实测量温度...")
    real_temps = extract_real_temps(cfg.real_data_csv, sensor_names, pred_times)
    
    # 5. 计算指标
    print("\n[5] 计算评价指标...")
    metrics = calculate_metrics(pred_temps, real_temps, sensor_names)
    
    # 6. 输出结果
    print("\n" + "=" * 70)
    print("评价指标")
    print("=" * 70)
    
    print(f"\n【{metrics['总体指标']}】")
    for key, value in metrics['总体指标'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\n【各传感器详细指标】")
    for sensor_name, sensor_metrics in metrics['各传感器指标'].items():
        print(f"\n  {sensor_name}:")
        for key, value in sensor_metrics.items():
            print(f"    {key}: {value:.4f}")
    
    # 7. 保存结果
    print(f"\n[6] 保存结果到 {cfg.output_dir}/...")
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # 保存指标到JSON
    import json
    with open(os.path.join(cfg.output_dir, f"sensor_metrics_{cfg.rpm}.json"), 'w', 
              encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  保存: sensor_metrics_{cfg.rpm}.json")
    
    # 保存详细数据到CSV
    results_df = pd.DataFrame({
        '时间(s)': pred_times,
    })
    for i, name in enumerate(sensor_names):
        results_df[f'{name}_预测'] = pred_temps[:, i]
        results_df[f'{name}_真实'] = real_temps[:, i]
        results_df[f'{name}_误差'] = pred_temps[:, i] - real_temps[:, i]
    
    results_df.to_csv(os.path.join(cfg.output_dir, f"sensor_comparison_{cfg.rpm}.csv"), 
                     index=False, encoding='utf-8-sig')
    print(f"  保存: sensor_comparison_{cfg.rpm}.csv")
    
    # 8. 绘制图表
    print("\n[7] 绘制对比图表...")
    plot_comparison(sensor_names, pred_times, pred_temps, real_temps, 
                   cfg.output_dir, cfg.rpm)
    
    print("\n" + "=" * 70)
    print("验证完成！")
    print(f"所有结果已保存到: {cfg.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()

