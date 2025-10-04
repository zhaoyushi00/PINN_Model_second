"""检查训练数据的升温过程"""
import pandas as pd
import numpy as np

def read_csv_auto(path):
    for enc in ["utf-8", "utf-8-sig", "gbk", "ansi"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except:
            continue
    return pd.read_csv(path)

print("="*70)
print("检查 9000 RPM 训练数据的前60秒升温过程")
print("="*70)

print(f"\n{'时间':<6} {'平均温度':<10} {'最小温度':<10} {'最大温度':<10} {'温度范围':<10}")
print("-"*60)

temps_all = []
for i in range(1, 61):
    df = read_csv_auto(f'data/snapshots/9000_csv/temp_{i:04d}.csv')
    
    # 找到温度列
    temp_col = None
    for col in ['Temperature (°C)', 'Temperature (℃)', 'Temperature']:
        if col in df.columns:
            temp_col = col
            break
    
    if temp_col is None:
        temp_col = df.columns[-1]  # 最后一列
    
    temps = df[temp_col].values
    temps_all.append(temps.mean())
    
    if i <= 12 or i % 10 == 0:
        print(f"{i:3d}秒  {temps.mean():7.2f}°C  {temps.min():7.2f}°C  {temps.max():7.2f}°C  {temps.max()-temps.min():7.2f}°C")

print("\n分析：")
print(f"  第1秒平均温度: {temps_all[0]:.2f}°C")
print(f"  第60秒平均温度: {temps_all[59]:.2f}°C")
print(f"  温升: {temps_all[59] - temps_all[0]:.2f}°C")

# 检查是否是平滑升温
is_smooth = all(temps_all[i+1] >= temps_all[i] for i in range(59))
print(f"  是否平滑升温: {'✅ 是' if is_smooth else '❌ 否'}")

# 计算POD系数
print("\n加载POD基...")
Phi = np.load("model/pod_phi.npy").astype(np.float32)
Tbar = np.load("model/pod_mean.npy").astype(np.float32)

print("\n检查前10秒的POD系数:")
print(f"{'时间':<6} {'a0':<10} {'a1':<10} {'a2':<10} {'a3':<10} {'a4':<10}")
print("-"*60)

for i in [1, 2, 3, 5, 10, 20, 60]:
    df = read_csv_auto(f'data/snapshots/9000_csv/temp_{i:04d}.csv')
    temp_col = None
    for col in ['Temperature (°C)', 'Temperature (℃)', 'Temperature']:
        if col in df.columns:
            temp_col = col
            break
    if temp_col is None:
        temp_col = df.columns[-1]
    
    T = df[temp_col].values.astype(np.float32)
    a = Phi.T @ (T - Tbar)
    
    print(f"{i:3d}秒  {a[0]:>9.2f} {a[1]:>9.2f} {a[2]:>9.2f} {a[3]:>9.2f} {a[4]:>9.2f}")

print("\n"+"="*70)
print("结论：")
print("  如果看到前60秒温度平滑上升，说明训练数据确实包含完整的冷启动过程")
print("  a0_net应该能从这些数据中学习正确的初始状态")
print("="*70)

