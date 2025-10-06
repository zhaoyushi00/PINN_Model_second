"""
物理特征工程模块
用于处理生热率、换热系数等物理参数的变异性和泛化性
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
from dataclasses import dataclass


@dataclass
class PhysicsFeatureConfig:
    """物理特征工程配置"""
    # 基准工况定义
    base_rpm: int = 9000
    base_ambient: float = 20.0

    # 物理参数基准值（从meta中获取）
    base_Q_stator: float = 98395.0
    base_Q_rotor: float = 189718.0
    base_h_housing: float = 9.7
    base_h_cooling: float = 191.26

    # 变异性范围（工程经验值）
    Q_variation_range: Tuple[float, float] = (0.85, 1.15)  # ±15%
    h_variation_range: Tuple[float, float] = (0.9, 1.1)   # ±10%

    # 环境温度影响系数
    ambient_impact_Q: float = 0.02    # 每度环境温差对热源的影响
    ambient_impact_h: float = 0.005   # 每度环境温差对换热的影响


class PhysicsFeatureEngineer:
    """物理特征工程器：处理物理参数变异性和泛化性"""

    def __init__(self, config: PhysicsFeatureConfig = None):
        self.config = config or PhysicsFeatureConfig()
        self._fit_scalers()

    def _fit_scalers(self):
        """拟合标准化器"""
        # 基于基准值计算标准化参数
        self.Q_stator_mean = self.config.base_Q_stator
        self.Q_stator_std = self.config.base_Q_stator * 0.15  # 假设15%变异

        self.Q_rotor_mean = self.config.base_Q_rotor
        self.Q_rotor_std = self.config.base_Q_rotor * 0.15

        self.h_housing_mean = self.config.base_h_housing
        self.h_housing_std = self.config.base_h_housing * 0.1

    def compute_derived_physics_features(self, rpm: float, ambient: float) -> Dict[str, float]:
        """基于工程公式计算派生物理特征"""

        # 1. 基础热源计算（幂律关系）
        Q_stator_base = 4.5e-8 * rpm**2.0  # 从拟合结果调整
        Q_rotor_base = 8.7e-8 * rpm**2.0

        # 2. 环境温度修正
        ambient_factor_Q = 1.0 + self.config.ambient_impact_Q * (ambient - self.config.base_ambient)
        ambient_factor_h = 1.0 + self.config.ambient_impact_h * (ambient - self.config.base_ambient)

        Q_stator_corrected = Q_stator_base * ambient_factor_Q
        Q_rotor_corrected = Q_rotor_base * ambient_factor_Q
        h_housing_corrected = self.config.base_h_housing * ambient_factor_h

        # 3. 派生复合特征
        total_heat_generation = Q_stator_corrected + Q_rotor_corrected
        total_heat_dissipation = h_housing_corrected * 1000  # 假设散热面积1000m2
        thermal_load_ratio = total_heat_generation / (total_heat_dissipation + 1e-6)

        # 4. 相对特征（增强泛化性）
        Q_stator_ratio = Q_stator_corrected / self.config.base_Q_stator
        Q_rotor_ratio = Q_rotor_corrected / self.config.base_Q_rotor
        h_housing_ratio = h_housing_corrected / self.config.base_h_housing

        return {
            'Q_stator_corrected': Q_stator_corrected,
            'Q_rotor_corrected': Q_rotor_corrected,
            'h_housing_corrected': h_housing_corrected,
            'total_heat_generation': total_heat_generation,
            'total_heat_dissipation': total_heat_dissipation,
            'thermal_load_ratio': thermal_load_ratio,
            'Q_stator_ratio': Q_stator_ratio,
            'Q_rotor_ratio': Q_rotor_ratio,
            'h_housing_ratio': h_housing_ratio,
            'heat_balance_factor': total_heat_generation / (total_heat_dissipation * thermal_load_ratio + 1e-6)
        }

    def add_parameter_variability(self, features: Dict[str, float],
                                 variation_level: str = 'normal') -> Dict[str, float]:
        """为物理参数添加工程变异性"""

        # 定义变异水平
        variation_multipliers = {
            'low': (0.95, 1.05),      # ±5%
            'normal': (0.90, 1.10),   # ±10%
            'high': (0.85, 1.15)      # ±15%
        }

        var_range = variation_multipliers.get(variation_level, variation_multipliers['normal'])

        # 为关键参数添加随机变异
        varied_features = features.copy()

        # 热源变异（制造差异、老化等）
        Q_stator_noise = np.random.uniform(var_range[0], var_range[1])
        Q_rotor_noise = np.random.uniform(var_range[0], var_range[1])

        varied_features['Q_stator_corrected'] *= Q_stator_noise
        varied_features['Q_rotor_corrected'] *= Q_rotor_noise

        # 换热系数变异（污垢、磨损等）
        h_housing_noise = np.random.uniform(var_range[0], var_range[1])
        varied_features['h_housing_corrected'] *= h_housing_noise

        # 重新计算复合特征
        varied_features['total_heat_generation'] = (
            varied_features['Q_stator_corrected'] + varied_features['Q_rotor_corrected']
        )
        varied_features['total_heat_dissipation'] = varied_features['h_housing_corrected'] * 1000
        varied_features['thermal_load_ratio'] = (
            varied_features['total_heat_generation'] /
            (varied_features['total_heat_dissipation'] + 1e-6)
        )

        return varied_features

    def normalize_physics_features(self, features: Dict[str, float]) -> torch.Tensor:
        """标准化物理特征"""

        # 选择需要标准化的特征
        normalize_keys = [
            'Q_stator_ratio', 'Q_rotor_ratio', 'h_housing_ratio',
            'thermal_load_ratio', 'heat_balance_factor'
        ]

        normalized_values = []
        for key in normalize_keys:
            if key in features:
                # Z-score标准化
                value = features[key]
                if key.startswith('Q_') and key.endswith('_ratio'):
                    mean = getattr(self, f"{key.replace('_ratio', '')}_mean", 1.0)
                    std = getattr(self, f"{key.replace('_ratio', '')}_std", 0.1)
                else:
                    mean = getattr(self, f"{key}_mean", 0.0)
                    std = getattr(self, f"{key}_std", 1.0)

                normalized = (value - mean) / (std + 1e-8)
                normalized_values.append(normalized)

        return torch.tensor(normalized_values, dtype=torch.float32)

    def create_domain_labels(self, features: Dict[str, float]) -> str:
        """基于物理特征创建域标签（用于域适应）"""

        thermal_load = features['thermal_load_ratio']

        if thermal_load < 0.3:
            return 'light_thermal_load'
        elif thermal_load < 0.7:
            return 'medium_thermal_load'
        else:
            return 'heavy_thermal_load'

    def process_batch_physics_features(self, rpm_batch: torch.Tensor,
                                     ambient_batch: torch.Tensor,
                                     variation_level: str = 'normal') -> Tuple[torch.Tensor, List[str]]:
        """批量处理物理特征"""

        batch_size = rpm_batch.size(0)
        all_features = []
        domain_labels = []

        for i in range(batch_size):
            rpm = rpm_batch[i].item()
            ambient = ambient_batch[i].item()

            # 计算派生物理特征
            features = self.compute_derived_physics_features(rpm, ambient)

            # 添加变异性
            if variation_level != 'none':
                features = self.add_parameter_variability(features, variation_level)

            # 标准化
            normalized = self.normalize_physics_features(features)
            all_features.append(normalized)

            # 域标签
            domain_labels.append(self.create_domain_labels(features))

        return torch.stack(all_features), domain_labels


def test_physics_feature_engineer():
    """测试物理特征工程器"""
    config = PhysicsFeatureConfig()
    engineer = PhysicsFeatureEngineer(config)

    # 测试不同工况
    test_cases = [
        (9000, 20.0),
        (12000, 25.0),
        (15000, 30.0),
        (18000, 15.0)
    ]

    print("物理特征工程测试：")
    print("=" * 60)

    for rpm, ambient in test_cases:
        features = engineer.compute_derived_physics_features(rpm, ambient)
        varied_features = engineer.add_parameter_variability(features, 'normal')
        normalized = engineer.normalize_physics_features(varied_features)
        domain_label = engineer.create_domain_labels(varied_features)

        print(f"\n工况: RPM={rpm}, Ambient={ambient}°C")
        print(f"  派生特征: Q_stator={features['Q_stator_corrected']:.0f}W, "
              f"Q_rotor={features['Q_rotor_corrected']:.0f}W, "
              f"h_housing={features['h_housing_corrected']:.2f}W/m2K")
        print(f"  热负荷比: {features['thermal_load_ratio']:.3f}")
        print(f"  标准化特征: {normalized}")
        print(f"  域标签: {domain_label}")

        # 显示变异幅度
        q_stator_change = (varied_features['Q_stator_corrected'] / features['Q_stator_corrected'] - 1) * 100
        print(f"  变异幅度: Q_stator {q_stator_change:.1f}%")


if __name__ == "__main__":
    test_physics_feature_engineer()
