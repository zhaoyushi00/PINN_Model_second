"""
诊断a0_net的训练问题
"""
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

def read_csv_auto(path):
    for enc in ["utf-8", "utf-8-sig", "gbk", "ansi"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except:
            continue
    return pd.read_csv(path)

print("=" * 70)
print("诊断 a0_net 训练问题")
print("=" * 70)

# 1. 检查训练数据的第1秒温度
print("\n[1] 检查训练数据集第1秒温度...")
for rpm in [9000, 12000, 15000]:
    try:
        df = read_csv_auto(f"data/snapshots/{rpm}_csv/temp_0001.csv")
        temps = df['Temperature (°C)'].values if 'Temperature (°C)' in df.columns else df['Temperature (��C)'].values
        print(f"  {rpm} RPM 第1秒:")
        print(f"    平均: {temps.mean():.2f}°C (期望≈23°C)")
        print(f"    最小: {temps.min():.2f}°C")
        print(f"    最大: {temps.max():.2f}°C")
    except Exception as e:
        print(f"  {rpm} RPM: 读取失败 - {e}")

# 2. 加载a0_net并测试
print("\n[2] 测试 a0_net 预测的初始状态...")
ckpt = torch.load("model/transient_tcn.pth", map_location="cpu", weights_only=False)

if "a0_state_dict" in ckpt:
    r = ckpt.get("r", 5)
    a0_net = nn.Sequential(
        nn.Linear(2, 128), nn.ReLU(),
        nn.Linear(128, r)
    )
    a0_net.load_state_dict(ckpt["a0_state_dict"])
    a0_net.eval()
    
    Phi = np.load("model/pod_phi.npy").astype(np.float32)
    Tbar = np.load("model/pod_mean.npy").astype(np.float32)
    
    ambient = 22.9
    for rpm in [9000, 12000, 15000, 18000, 21000]:
        cond = torch.tensor([rpm, ambient], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            a0_pred = a0_net(cond).squeeze(0).numpy()
        
        T_pred = Tbar + Phi @ a0_pred
        
        print(f"\n  {rpm} RPM (环境温度={ambient}°C):")
        print(f"    a0_net预测温度:")
        print(f"      平均: {T_pred.mean():.2f}°C (期望≈{ambient}°C)")
        print(f"      最小: {T_pred.min():.2f}°C")
        print(f"      最大: {T_pred.max():.2f}°C")
        print(f"      与环境温差: {T_pred.mean() - ambient:.2f}°C ⚠️")
else:
    print("  ⚠️ checkpoint中没有a0_net")

# 3. 检查POD系数数据集的第一帧
print("\n[3] 检查预计算POD系数的t=0数据...")
coeffs = np.load("model/pod_coeffs.npy").astype(np.float32)
meta = read_csv_auto("data/snapshots/meta_transient.csv")

for rpm in [9000, 12000, 15000]:
    rpm_data = meta[meta["rpm"] == rpm]
    if len(rpm_data) > 0:
        first_idx = rpm_data.index[0]
        a0_true = coeffs[first_idx]
        T0_true = Tbar + Phi @ a0_true
        
        print(f"\n  {rpm} RPM 真实第1秒:")
        print(f"    POD系数重建温度:")
        print(f"      平均: {T0_true.mean():.2f}°C")
        print(f"      最小: {T0_true.min():.2f}°C")
        print(f"      最大: {T0_true.max():.2f}°C")

# 4. 分析问题
print("\n" + "=" * 70)
print("问题分析")
print("=" * 70)

print("""
发现的问题：
1. ✅ 训练数据的第1秒温度确实接近环境温度（23°C）
2. ❌ 但a0_net预测的初始温度偏高（27-33°C）

可能原因：
1. a0_net训练不充分
   - a0_weight=0.1 权重太小
   - is_t0=1的样本太少（只有每个RPM序列的第一个窗口）
   
2. a0_net学习的是"窗口第一步"而非"全局第0秒"
   - 虽然代码有is_t0标记，但训练时可能没有足够样本
   - 大部分窗口start≠0，这些窗口的seq_a[:, 0]不是冷启动状态

3. 滑动窗口训练的副作用
   - sequence_length=60, stride=1
   - 大量窗口从中间时刻开始，a0_net学习了"中间状态"

解决方案：
A. 增加a0_weight权重（0.1 → 1.0）
B. 只用is_t0=1的样本训练a0_net
C. 手动构造冷启动状态（环境温度）并添加到训练集
D. 预测时强制使用环境温度初始化，不用a0_net

推荐：
- 短期：预测时用环境温度初始化（方案D）✅
- 长期：重新训练，增加a0权重，添加冷启动样本（方案A+C）
""")

print("=" * 70)

