# main.py
import torch
from pathlib import Path

from build_pod import build_pod_from_folder
from prepare_transient_meta import build_transient_meta
from train_transient import TrainConfig, train as train_transient

# =====================
# 基础配置（统一管理）
# =====================
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
FORCE_REBUILD_POD  = True        # 强制重建POD以使用新配置

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


def main():
    if torch.cuda.is_available():
        device = "cuda"
        print(">>> Using device: CUDA")
        print(f">>> CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print(">>> Using device: CPU (CUDA not available)")

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
        coeffs_path=str(coeffs_path),
        coords_path=str(coords_path),
    )
    train_transient(cfg)

    print("\n瞬态流程完成！模型已保存到 model/ 目录")


if __name__ == "__main__":
    main()
