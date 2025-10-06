"""
XGBoost残差纠偏系统
学习GRU预测的残差模式，进行一步纠偏
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple, Optional
import torch
import joblib
import os


class XGBoostResidualCorrector:
    """XGBoost残差纠偏器"""

    def __init__(self,
                 n_estimators: int = 500,
                 max_depth: int = 6,
                 learning_rate: float = 0.05,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 random_state: int = 42,
                 residual_window_size: int = 20,
                 feature_cache_size: int = 1000):
        """
        Args:
            n_estimators: XGBoost树的数量
            max_depth: 树的最大深度
            learning_rate: 学习率
            subsample: 子采样比例
            colsample_bytree: 特征子采样比例
            random_state: 随机种子
            residual_window_size: 残差窗口大小（用于特征工程）
            feature_cache_size: 特征缓存大小（内存优化）
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.residual_window_size = residual_window_size
        self.feature_cache_size = feature_cache_size

        # 初始化XGBoost模型
        self.base_model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            verbosity=0
        )
        self.model = MultiOutputRegressor(self.base_model)

        # 特征缩放器
        self.feature_scaler = StandardScaler()

        # 训练状态
        self.is_fitted = False
        self.feature_names = []

        # 特征缓存（用于在线更新）
        self.feature_cache = []
        self.residual_cache = []

    def extract_residual_features(self,
                                 gru_predictions: np.ndarray,
                                 true_values: np.ndarray,
                                 conditions: np.ndarray,
                                 seq_windows: Optional[np.ndarray] = None) -> np.ndarray:
        """
        提取残差特征
        Args:
            gru_predictions: GRU预测值 [N, output_dim]
            true_values: 真实值 [N, output_dim]
            conditions: 工况条件 [N, 2] (rpm, ambient)
            seq_windows: 序列窗口（可选） [N, window_size, input_dim]
        """
        N, output_dim = gru_predictions.shape

        features = []

        # 1. 基础残差特征
        residuals = true_values - gru_predictions  # [N, output_dim]

        # 当前残差
        features.append(residuals)

        # 残差统计特征
        residual_mean = np.mean(residuals, axis=1, keepdims=True)
        residual_std = np.std(residuals, axis=1, keepdims=True)
        residual_max = np.max(np.abs(residuals), axis=1, keepdims=True)

        features.extend([residual_mean, residual_std, residual_max])

        # 2. GRU预测特征
        gru_mean = np.mean(gru_predictions, axis=1, keepdims=True)
        gru_std = np.std(gru_predictions, axis=1, keepdims=True)
        gru_trend = gru_predictions[:, -1:] - gru_predictions[:, 0:1]  # 趋势

        features.extend([gru_mean, gru_std, gru_trend])

        # 3. 工况特征
        rpm = conditions[:, 0:1]
        ambient = conditions[:, 1:2]

        # 工况相对值（相对于基准）
        rpm_relative = rpm / 9000.0
        ambient_relative = (ambient - 20.0) / 10.0

        features.extend([rpm_relative, ambient_relative])

        # 4. 序列特征（如果提供）
        if seq_windows is not None:
            # 序列统计
            seq_mean = np.mean(seq_windows, axis=1)  # [N, input_dim]
            seq_std = np.std(seq_windows, axis=1)
            seq_trend = seq_windows[:, -1] - seq_windows[:, 0]  # [N, input_dim]

            # 序列变化率
            seq_diff = np.diff(seq_windows, axis=1)  # [N, window_size-1, input_dim]
            seq_change_rate = np.mean(seq_diff, axis=1)  # [N, input_dim]

            features.extend([seq_mean, seq_std, seq_trend, seq_change_rate])

        # 拼接所有特征
        feature_matrix = np.concatenate(features, axis=1)

        # 添加特征名称（用于调试）
        self.feature_names = self._generate_feature_names(output_dim, seq_windows is not None)

        return feature_matrix

    def _generate_feature_names(self, output_dim: int, use_sequence_features: bool) -> List[str]:
        """生成特征名称"""
        names = []

        # 残差特征
        for i in range(output_dim):
            names.append(f'residual_{i}')
        names.append('residual_mean')
        names.append('residual_std')
        names.append('residual_max')

        # GRU预测特征
        for i in range(output_dim):
            names.append(f'gru_pred_{i}')
        names.append('gru_mean')
        names.append('gru_std')
        names.append('gru_trend')

        # 工况特征
        names.extend(['rpm_relative', 'ambient_relative'])

        # 序列特征
        if use_sequence_features:
            for i in range(output_dim):
                names.extend([f'seq_mean_{i}', f'seq_std_{i}', f'seq_trend_{i}', f'seq_change_{i}'])

        return names

    def fit(self,
            gru_predictions: np.ndarray,
            true_values: np.ndarray,
            conditions: np.ndarray,
            seq_windows: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        训练残差模型
        Args:
            gru_predictions: GRU预测值
            true_values: 真实值
            conditions: 工况条件
            seq_windows: 序列窗口
            sample_weight: 样本权重（可选）
        """
        # 提取特征
        X = self.extract_residual_features(gru_predictions, true_values, conditions, seq_windows)

        # 计算残差作为目标
        y = true_values - gru_predictions

        # 特征标准化
        X_scaled = self.feature_scaler.fit_transform(X)

        # 训练模型
        if sample_weight is not None:
            # XGBoost不支持样本权重，需要手动实现
            self.model.fit(X_scaled, y)
        else:
            self.model.fit(X_scaled, y)

        self.is_fitted = True

        # 计算训练指标
        y_pred = self.model.predict(X_scaled)
        train_metrics = {
            'train_mae': mean_absolute_error(y, y_pred),
            'train_rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'train_r2': r2_score(y, y_pred),
            'n_samples': len(X),
            'n_features': X.shape[1]
        }

        print(f"残差模型训练完成 - MAE: {train_metrics['train_mae']:.4f}, "
              f"RMSE: {train_metrics['train_rmse']:.4f}, R2: {train_metrics['train_r2']:.4f}")

        return train_metrics

    def predict(self,
               gru_predictions: np.ndarray,
               conditions: np.ndarray,
               seq_windows: Optional[np.ndarray] = None) -> np.ndarray:
        """
        预测残差并纠偏
        Args:
            gru_predictions: GRU预测值
            conditions: 工况条件
            seq_windows: 序列窗口
        """
        if not self.is_fitted:
            print("警告：残差模型尚未训练，返回原始GRU预测")
            return gru_predictions

        # 提取特征
        X = self.extract_residual_features(gru_predictions, gru_predictions, conditions, seq_windows)

        # 特征标准化（使用训练时的缩放器）
        X_scaled = self.feature_scaler.transform(X)

        # 预测残差
        residual_pred = self.model.predict(X_scaled)

        # 纠偏
        corrected_predictions = gru_predictions + residual_pred

        return corrected_predictions

    def update_online(self,
                     gru_predictions: np.ndarray,
                     true_values: np.ndarray,
                     conditions: np.ndarray,
                     seq_windows: Optional[np.ndarray] = None,
                     learning_rate: float = 0.1) -> Dict[str, float]:
        """
        在线更新残差模型（增量学习）
        Args:
            gru_predictions: GRU预测值
            true_values: 真实值
            conditions: 工况条件
            seq_windows: 序列窗口
            learning_rate: 在线学习率
        """
        if not self.is_fitted:
            print("残差模型尚未初始化，进行完整训练")
            return self.fit(gru_predictions, true_values, conditions, seq_windows)

        # 提取特征和残差
        X = self.extract_residual_features(gru_predictions, true_values, conditions, seq_windows)
        y = true_values - gru_predictions

        # 标准化特征
        X_scaled = self.feature_scaler.transform(X)

        # 增量学习（简化版：定期重新训练）
        # 实际应用中可以使用XGBoost的incremental learning或定期全量更新

        # 添加到缓存
        self.feature_cache.append(X_scaled)
        self.residual_cache.append(y)

        # 定期更新模型
        if len(self.feature_cache) >= self.feature_cache_size:
            print("缓存满，更新残差模型...")

            # 合并缓存数据
            X_all = np.vstack(self.feature_cache)
            y_all = np.vstack(self.residual_cache)

            # 重新训练模型（可以考虑使用部分历史数据）
            self.model.fit(X_all, y_all)

            # 清空缓存
            self.feature_cache = []
            self.residual_cache = []

            # 计算更新后指标
            y_pred = self.model.predict(X_all)
            update_metrics = {
                'update_mae': mean_absolute_error(y_all, y_pred),
                'update_rmse': np.sqrt(mean_squared_error(y_all, y_pred)),
                'update_r2': r2_score(y_all, y_pred),
                'n_update_samples': len(X_all)
            }

            print(f"残差模型更新完成 - MAE: {update_metrics['update_mae']:.4f}, "
                  f"RMSE: {update_metrics['update_rmse']:.4f}")

            return update_metrics

        return {}

    def save_model(self, filepath: str):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'model': self.model,
            'feature_scaler': self.feature_scaler,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'config': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'residual_window_size': self.residual_window_size
            }
        }
        joblib.dump(model_data, filepath)
        print(f"残差模型已保存到: {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        if not os.path.exists(filepath):
            print(f"模型文件不存在: {filepath}")
            return False

        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_scaler = model_data['feature_scaler']
        self.is_fitted = model_data['is_fitted']
        self.feature_names = model_data['feature_names']

        # 更新配置
        config = model_data.get('config', {})
        for key, value in config.items():
            setattr(self, key, value)

        print(f"残差模型已加载: {filepath}")
        return True

    def compute_importance_scores(self) -> Dict[str, float]:
        """计算特征重要性"""
        if not self.is_fitted:
            return {}

        # 获取基础模型的特征重要性
        base_model = self.model.estimators_[0]  # 第一个输出维度的模型
        importance = base_model.feature_importances_

        # 转换为字典
        importance_dict = {}
        for i, name in enumerate(self.feature_names[:len(importance)]):
            importance_dict[name] = float(importance[i])

        # 排序
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        return sorted_importance


def test_xgboost_residual():
    """测试XGBoost残差纠偏器"""
    # 生成模拟数据
    np.random.seed(42)
    N = 1000
    input_dim = 5
    output_dim = 5

    # 模拟GRU预测（带系统性偏差）
    gru_pred = np.random.randn(N, output_dim) * 0.5
    true_values = gru_pred + np.random.randn(N, output_dim) * 0.1 + 0.2  # 添加系统偏差

    # 工况条件
    conditions = np.random.uniform([8000, 15], [22000, 35], (N, 2))

    # 序列窗口（简化）
    seq_windows = np.random.randn(N, 10, input_dim)

    # 创建纠偏器
    corrector = XGBoostResidualCorrector(
        n_estimators=100,  # 减少数量用于测试
        max_depth=4,
        learning_rate=0.1
    )

    print("XGBoost残差纠偏测试：")
    print("=" * 50)

    # 训练
    print("训练残差模型...")
    train_metrics = corrector.fit(gru_pred, true_values, conditions, seq_windows)
    print(f"训练指标: {train_metrics}")

    # 预测纠偏
    print("\n进行残差纠偏...")
    corrected_pred = corrector.predict(gru_pred, conditions, seq_windows)

    # 计算纠偏效果
    original_mae = mean_absolute_error(true_values, gru_pred)
    corrected_mae = mean_absolute_error(true_values, corrected_pred)

    print(f"纠偏效果:")
    print(f"  原始MAE: {original_mae:.4f}")
    print(f"  纠偏后MAE: {corrected_mae:.4f}")
    print(f"  改善幅度: {(original_mae - corrected_mae) / original_mae * 100:.2f}%")

    # 特征重要性
    importance = corrector.compute_importance_scores()
    print(f"\n特征重要性 (Top 5):")
    for i, (name, score) in enumerate(list(importance.items())[:5]):
        print(f"  {i+1}. {name}: {score:.4f}")

    return corrector


if __name__ == "__main__":
    test_xgboost_residual()
