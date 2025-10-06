# main.py
"""
热仿真PINN模型训练主入口
支持两种工程架构：
1. 【原架构】Randomized SVD + TCN + PINN（传统工程）
2. 【新架构】Randomized SVD + GRU + Rollout + Scheduled Sampling + XGBoost残差纠偏 + 物理特征工程（增强泛化工程）

快速使用指南：
   方式1：修改配置文件
       将第25行的 MODEL_ARCHITECTURE 改为 "traditional" 或 "enhanced"

   方式2：命令行参数
       python main.py --architecture traditional    # 使用TCN
       python main.py --architecture enhanced      # 使用GRU（默认）

   方式3：查看对比
       python main.py --show_comparison
"""
import torch
import argparse
import os
from pathlib import Path

from build_pod import build_pod_from_folder
from prepare_transient_meta import build_transient_meta
from train_transient import TrainConfig, train as train_transient
from train_gru_physics import GRUTrainConfig, train as train_gru_physics

# =====================
# 基础配置（统一管理）
# =====================

# 设置为 "traditional" 使用原TCN架构
# 设置为 "enhanced" 使用新GRU架构
# 也可以通过命令行参数 --architecture 覆盖此设置
MODEL_ARCHITECTURE = "enhanced"  # "traditional" 或 "enhanced"

# 路径
SNAPSHOTS_DIR      = "data/snapshots"
META_BASE_PATH     = f"{SNAPSHOTS_DIR}/meta.csv"
META_TRANSIENT_PATH= f"{SNAPSHOTS_DIR}/meta_transient.csv"
POD_SAVE_DIR       = "model"
SENSORS_MAP_CSV    = "data/sensors_map.csv"
REAL_STEADY_DIR    = "data/real_steady"

# 元信息生成
DELTA_T_SECONDS    = 1.0
FORCE_REBUILD_META = False

# POD 构建参数
POD_R              = None        # None 表示按能量保持率自动截断
POD_ENERGY_KEEP    = 0.99        # 降低到99%以保留更多瞬态模态（原0.999太高导致只保留2个）
POD_FRAME_STRIDE   = 1           # 每2秒采样1帧（保留50%瞬态细节，平衡内存与精度）
POD_MAX_PER_RPM    = None        # 使用全部帧（Randomized SVD内存友好，可处理18000帧）
POD_PROGRESS_EVERY = 200
POD_DTYPE          = "float32"   # float32节省50%内存
FORCE_REBUILD_POD  = False        # 强制重建POD以使用新配置

# TCN 训练参数
SEQ_LEN            = 240         # 增加到240秒历史（原60太短，学不到长期规律）
HORIZON            = 12          # 增加到12秒预测（原1太短，无法约束长期演化）
BATCH_SIZE         = 2           # 保持2（显存限制）
MAX_EPOCHS         = 50
LEARNING_RATE      = 1e-3
PHYSICS_WEIGHT     = 5e-5        # 稍微降低物理权重，等归一化后再调整
PHYSICS_KNN        = 8
TIME_STEP          = 1.0
SENSOR_LOSS_WEIGHT = 50.0        # 大幅增加传感器权重（原10太小）
VEL_LOSS_WEIGHT    = 1.0         # 增加速度正则权重（原0.1太小）
A0_WEIGHT          = 0.5         # 增加初始化权重（原0.1太小）
PHYSICS_MAX_NODES  = 5000
USE_PRECOMPUTED    = True


def print_architecture_comparison():
    """打印两种架构对比"""
    print("=" * 80)
    print("热仿真PINN模型训练 - 架构对比")
    print("=" * 80)

    print("\n【原架构】Randomized SVD + TCN + PINN（传统工程）")
    print("   核心特点：")
    print("   • 主干网络：TCN（时序卷积网络）")
    print("   • 物理约束：PINN物理残差正则")
    print("   • 训练策略：单步预测 + 教师强制")
    print("   • 优势：计算稳定，易于收敛")
    print("   • 劣势：泛化性有限，长时预测漂移")

    print("\n【新架构】Randomized SVD + GRU + Rollout + Scheduled Sampling + XGBoost残差纠偏 + 物理特征工程（增强泛化工程）")
    print("   核心特点：")
    print("   • 主干网络：GRU（门控循环单元）+ 物理特征输入")
    print("   • 训练策略：Rollout多步预测 + Scheduled Sampling退火")
    print("   • 纠偏机制：XGBoost残差学习")
    print("   • 物理增强：派生特征工程 + 域适应")
    print("   • 优势：强泛化性，长时稳定性好")
    print("   • 劣势：训练复杂度高，计算资源需求大")

    print("\n⚙️  参数配置对比：")
    print("   【原架构参数】（传统工程）")
    print(f"   • SEQ_LEN: {SEQ_LEN} (历史序列长度)")
    print(f"   • HORIZON: {HORIZON} (预测步长)")
    print(f"   • PHYSICS_WEIGHT: {PHYSICS_WEIGHT} (物理残差权重)")
    print(f"   • SENSOR_LOSS_WEIGHT: {SENSOR_LOSS_WEIGHT} (传感器损失权重)")

    print("\n   【新架构参数】（增强泛化工程）")
    print("   • SEQ_LEN: 240 (历史序列长度，更长历史)")
    print("   • HORIZON: 12 (预测步长，多步预测)")
    print("   • ROLLOUT_HORIZON: 12 (Rollout步长)")
    print("   • SCHEDULED_SAMPLING_EPOCHS: 20 (教师强制退火周期)")
    print("   • PHYSICS_VARIATION_LEVEL: 'normal' (物理参数变异水平)")
    print("   • DOMAIN_ADAPTATION_WEIGHT: 0.1 (域适应损失权重)")

    print("=" * 80)


def run_traditional_engineering(device: str):
    """运行传统工程：Randomized SVD + TCN + PINN"""
    print("\n🔧 运行传统工程：Randomized SVD + TCN + PINN")
    print("=" * 50)

    # 1) 生成瞬态元信息（若不存在或强制重建）
    meta_path = Path(META_TRANSIENT_PATH)
    if FORCE_REBUILD_META or not meta_path.exists():
        print(">>> 生成瞬态元信息 meta_transient.csv")
        build_transient_meta(
            snapshots_dir=Path(SNAPSHOTS_DIR),
            base_meta_path=Path(META_BASE_PATH),
            output_path=meta_path,
            delta_t=DELTA_T_SECONDS,
        )
    else:
        print(f">>> 检测到已有 meta：{meta_path}，跳过生成")

    # 2) 构建 POD（若缺失或强制重建）
    phi_path    = Path(POD_SAVE_DIR) / "pod_phi.npy"
    mean_path   = Path(POD_SAVE_DIR) / "pod_mean.npy"
    coeffs_path = Path(POD_SAVE_DIR) / "pod_coeffs.npy"
    coords_path = Path(POD_SAVE_DIR) / "pod_coords.npy"

    need_build_pod = FORCE_REBUILD_POD or not (phi_path.exists() and mean_path.exists() and coeffs_path.exists() and coords_path.exists())

    if need_build_pod:
        print(">>> 基于瞬态快照构建 POD")
        build_pod_from_folder(
            snapshot_dir=SNAPSHOTS_DIR,
            save_dir=POD_SAVE_DIR,
            r=POD_R,
            energy_keep=POD_ENERGY_KEEP,
            meta_csv=meta_path.name,
            frame_stride=POD_FRAME_STRIDE,
            max_files_per_rpm=POD_MAX_PER_RPM,
            progress_every=POD_PROGRESS_EVERY,
            dtype=POD_DTYPE,
        )
    else:
        print(f">>> 检测到已有 POD 输出，跳过构建：\n    {phi_path}\n    {mean_path}\n    {coeffs_path}\n    {coords_path}")

    # 3) 训练瞬态 TCN 模型
    print(">>> 训练瞬态 TCN 模型")
    cfg = TrainConfig(
        device=device,
        sequence_length=SEQ_LEN,
        horizon=HORIZON,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        lr=LEARNING_RATE,
        physics_weight=PHYSICS_WEIGHT,
        physics_knn=PHYSICS_KNN,
        physics_max_nodes=PHYSICS_MAX_NODES,
        time_step=TIME_STEP,
        sensors_map_csv=SENSORS_MAP_CSV,
        real_steady_dir=REAL_STEADY_DIR,
        sensor_loss_weight=SENSOR_LOSS_WEIGHT,
        vel_loss_weight=VEL_LOSS_WEIGHT,
        a0_weight=A0_WEIGHT,
        use_precomputed=USE_PRECOMPUTED,
        phi_path=str(phi_path),
        mean_path=str(mean_path),
        coeffs_path=str(coeffs_path),
        coords_path=str(coords_path),
    )
    train_transient(cfg)

    print("\n传统工程完成！模型已保存到 model/ 目录")


def run_enhanced_generalization_engineering(device: str):
    """运行增强泛化工程：Randomized SVD + GRU + Rollout + Scheduled Sampling + XGBoost残差纠偏 + 物理特征工程"""
    print("\n运行增强泛化工程：GRU + Rollout + Scheduled Sampling + XGBoost残差纠偏 + 物理特征工程")
    print("=" * 70)

    # 1) 生成瞬态元信息（若不存在或强制重建）
    meta_path = Path(META_TRANSIENT_PATH)
    if FORCE_REBUILD_META or not meta_path.exists():
        print(">>> 生成瞬态元信息 meta_transient.csv")
        build_transient_meta(
            snapshots_dir=Path(SNAPSHOTS_DIR),
            base_meta_path=Path(META_BASE_PATH),
            output_path=meta_path,
            delta_t=DELTA_T_SECONDS,
        )
    else:
        print(f">>> 检测到已有 meta：{meta_path}，跳过生成")

    # 2) 构建 POD（若缺失或强制重建）
    phi_path    = Path(POD_SAVE_DIR) / "pod_phi.npy"
    mean_path   = Path(POD_SAVE_DIR) / "pod_mean.npy"
    coeffs_path = Path(POD_SAVE_DIR) / "pod_coeffs.npy"
    coords_path = Path(POD_SAVE_DIR) / "pod_coords.npy"

    need_build_pod = FORCE_REBUILD_POD or not (phi_path.exists() and mean_path.exists() and coeffs_path.exists() and coords_path.exists())

    if need_build_pod:
        print(">>> 基于瞬态快照构建 POD")
        build_pod_from_folder(
            snapshot_dir=SNAPSHOTS_DIR,
            save_dir=POD_SAVE_DIR,
            r=POD_R,
            energy_keep=POD_ENERGY_KEEP,
            meta_csv=meta_path.name,
            frame_stride=POD_FRAME_STRIDE,
            max_files_per_rpm=POD_MAX_PER_RPM,
            progress_every=POD_PROGRESS_EVERY,
            dtype=POD_DTYPE,
        )
    else:
        print(f">>> 检测到已有 POD 输出，跳过构建：\n    {phi_path}\n    {mean_path}\n    {coeffs_path}\n    {coords_path}")

    # 3) 训练GRU物理感知模型
    print(">>> 训练GRU物理感知模型（带Rollout、Scheduled Sampling、物理特征工程）")
    gru_cfg = GRUTrainConfig(
        device=device,
        sequence_length=240,        # 增强泛化工程参数
        horizon=12,                 # 多步预测
        batch_size=8,              # 更大的batch size用于稳定训练
        max_epochs=100,            # 更多训练轮数
        lr=1e-3,
        hidden_dim=128,            # GRU隐藏维度
        num_layers=2,              # GRU层数
        dropout=0.1,
        use_physics_features=True, # 启用物理特征工程
        use_domain_adaptation=True,# 启用域适应
        physics_variation_level='normal',  # 物理参数变异水平
        rollout_horizon=12,        # Rollout步长
        scheduled_sampling_epochs=20,     # 教师强制退火周期
        use_xgboost_residual=True, # 启用XGBoost残差纠偏
        residual_window_size=20,   # 残差窗口大小
        domain_loss_weight=0.1,    # 域适应损失权重
        phi_path=str(phi_path),
        mean_path=str(mean_path),
        coeffs_path=str(coeffs_path),
        coords_path=str(coords_path),
    )
    train_gru_physics(gru_cfg)

    print("\n增强泛化工程完成！模型已保存到 model/ 目录")


def main():
    """主函数：支持两种工程选择"""
    parser = argparse.ArgumentParser(description='热仿真PINN模型训练')
    parser.add_argument('--architecture', type=str, choices=['traditional', 'enhanced'],
                       help='选择工程架构：traditional（原TCN）或enhanced（新GRU）。如果不指定，使用MODEL_ARCHITECTURE配置')
    parser.add_argument('--show_comparison', action='store_true',
                       help='显示两种架构对比')

    args = parser.parse_args()

    # 优先级：命令行参数 > 配置文件变量 > 默认值
    if args.architecture:
        selected_architecture = args.architecture
    elif os.environ.get('MODEL_ARCHITECTURE'):
        selected_architecture = os.environ.get('MODEL_ARCHITECTURE')
    else:
        selected_architecture = MODEL_ARCHITECTURE

    print(f"选择架构: {selected_architecture}")
    if selected_architecture == "traditional":
        print(" 使用传统工程：Randomized SVD + TCN + PINN")
    elif selected_architecture == "enhanced":
        print(" 使用增强泛化工程：Randomized SVD + GRU + Rollout + Scheduled Sampling + XGBoost残差纠偏 + 物理特征工程")
    else:
        print(f"未知架构: {selected_architecture}")
        print("请设置 MODEL_ARCHITECTURE 为 'traditional' 或 'enhanced'")
        return

    # 设备检测
    if torch.cuda.is_available():
        device = "cuda"
        print(">>> Using device: CUDA")
        print(f">>> CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print(">>> Using device: CPU (CUDA not available)")

    # 显示架构对比（如果请求）
    if args.show_comparison:
        print_architecture_comparison()
        return

    # 根据选择运行相应工程
    if selected_architecture == 'traditional':
        run_traditional_engineering(device)
    elif selected_architecture == 'enhanced':
        run_enhanced_generalization_engineering(device)
    else:
        print(f"未知架构: {selected_architecture}")
        print("使用 --show_comparison 查看可用选项")


if __name__ == "__main__":
    main()
