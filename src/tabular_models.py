"""
表格数据模型兼容性 (Tabular Model Compatibility)
================================================
实现方向5：为表格数据（Adult, Covertype, Dry Bean）提供多种模型选择
- MLP (原有)
- XGBoost/LightGBM (新增，树模型更适合表格数据)
- TabNet (可选，深度表格学习)

参考：Lu et al. (TMLR 2025) 发现不确定性采样在表格数据上的有效性
强烈依赖于模型兼容性——树模型可能比MLP更适合不确定性估计。
"""

from typing import Optional, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TabularMLP(nn.Module):
    """标准 MLP，用于表格数据分类。"""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
        self._feature_dim = prev_dim

    def forward(self, x, return_features: bool = False):
        feat = self.features(x)
        logits = self.classifier(feat)
        if return_features:
            return logits, feat
        return logits

    def get_features(self, x):
        return self.features(x)


class XGBoostTabularWrapper:
    """
    XGBoost 包装器，使其接口与 PyTorch 模型兼容。
    用于表格数据的 AL 查询和训练。
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        use_gpu: bool = False,
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.model = None
        self._feature_dim = input_dim  # XGBoost 使用原始特征

    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练 XGBoost 模型。"""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost is required. Install with: pip install xgboost")

        params = {
            "objective": "multi:softprob" if self.num_classes > 2 else "binary:logistic",
            "eval_metric": "mlogloss" if self.num_classes > 2 else "logloss",
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
            "n_estimators": self.n_estimators,
            "verbosity": 0,
        }

        if self.num_classes > 2:
            params["num_class"] = self.num_classes

        if self.use_gpu:
            params["tree_method"] = "gpu_hist"
            params["predictor"] = "gpu_predictor"
        else:
            params["tree_method"] = "hist"

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率。"""
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        probs = self.model.predict_proba(X)
        # 二分类时 XGBoost 返回 (N, 2)，需要确保形状一致
        if self.num_classes == 2 and probs.ndim == 1:
            probs = np.column_stack([1 - probs, probs])
        return probs.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别。"""
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        return self.model.predict(X)

    def get_features(self, X: np.ndarray) -> np.ndarray:
        """获取特征（XGBoost 使用原始输入特征作为表示）。"""
        return X

    def to(self, device):
        """兼容 PyTorch 接口（XGBoost 忽略 device）。"""
        return self

    def eval(self):
        """兼容 PyTorch 接口。"""
        pass

    def train(self, mode=True):
        """兼容 PyTorch 接口。"""
        pass

    def state_dict(self):
        """获取模型状态。"""
        if self.model is None:
            return {}
        return {"model": self.model.get_booster().save_raw()}

    def load_state_dict(self, state_dict):
        """加载模型状态。"""
        if "model" in state_dict and self.model is not None:
            # 简化处理：重新训练
            pass


class LightGBMWrapper:
    """
    LightGBM 包装器。
    通常比 XGBoost 更快，且对表格数据效果相当。
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.model = None
        self._feature_dim = input_dim

    def fit(self, X: np.ndarray, y: np.ndarray):
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required. Install with: pip install lightgbm")

        params = {
            "objective": "multiclass" if self.num_classes > 2 else "binary",
            "metric": "multi_logloss" if self.num_classes > 2 else "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "feature_fraction": self.colsample_bytree,
            "bagging_fraction": self.subsample,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": self.random_state,
            "n_estimators": self.n_estimators,
        }

        if self.max_depth > 0:
            params["max_depth"] = self.max_depth

        if self.num_classes > 2:
            params["num_class"] = self.num_classes

        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        probs = self.model.predict_proba(X)
        if self.num_classes == 2 and probs.ndim == 1:
            probs = np.column_stack([1 - probs, probs])
        return probs.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        return self.model.predict(X)

    def get_features(self, X: np.ndarray) -> np.ndarray:
        return X

    def to(self, device):
        return self

    def eval(self):
        pass

    def train(self, mode=True):
        pass

    def state_dict(self):
        if self.model is None:
            return {}
        return {"model": self.model.booster_.model_to_string()}

    def load_state_dict(self, state_dict):
        pass


def extract_numpy_from_dataset(dataset: Dataset, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 PyTorch Dataset 中提取 numpy 数组（用于树模型）。

    Returns:
        (X, y): numpy 数组
    """
    X_list, y_list = [], []
    for idx in indices:
        item = dataset[idx]
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            x, y = item[0], item[1]
        else:
            x, y = item, None

        # 处理 tensor
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if isinstance(y, torch.Tensor):
            y = y.item()

        X_list.append(x)
        if y is not None:
            y_list.append(y)

    X = np.array(X_list)
    y = np.array(y_list) if y_list else None
    return X, y


def create_tabular_model(
    model_type: str,
    input_dim: int,
    num_classes: int,
    device: torch.device = None,
    **kwargs,
) -> Any:
    """
    工厂函数：创建表格数据模型。

    Args:
        model_type: "mlp" | "xgboost" | "lightgbm"
        input_dim: 输入特征维度
        num_classes: 类别数
        device: PyTorch 设备（对树模型无效）
        **kwargs: 额外参数

    Returns:
        模型实例
    """
    if model_type == "mlp":
        hidden_dims = kwargs.get("hidden_dims", [256, 128])
        dropout = kwargs.get("dropout", 0.2)
        model = TabularMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        if device is not None:
            model = model.to(device)
        return model

    elif model_type == "xgboost":
        return XGBoostTabularWrapper(
            input_dim=input_dim,
            num_classes=num_classes,
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.1),
            random_state=kwargs.get("random_state", 42),
        )

    elif model_type == "lightgbm":
        return LightGBMWrapper(
            input_dim=input_dim,
            num_classes=num_classes,
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", -1),
            learning_rate=kwargs.get("learning_rate", 0.1),
            random_state=kwargs.get("random_state", 42),
        )

    else:
        raise ValueError(f"Unknown tabular model type: {model_type}")


def train_tabular_model(
    model: Any,
    dataset: Dataset,
    train_idx: List[int],
    device: torch.device = None,
    n_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    **kwargs,
) -> float:
    """
    统一训练接口：支持 MLP（PyTorch）和树模型（XGBoost/LightGBM）。

    Returns:
        训练时间（秒）
    """
    import time

    t0 = time.time()

    if isinstance(model, (XGBoostTabularWrapper, LightGBMWrapper)):
        # 树模型训练
        X_train, y_train = extract_numpy_from_dataset(dataset, train_idx)
        model.fit(X_train, y_train)
        return time.time() - t0

    elif isinstance(model, TabularMLP):
        # PyTorch MLP 训练
        model.train()
        subset = torch.utils.data.Subset(dataset, train_idx)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(n_epochs):
            for batch_data in loader:
                if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 2:
                    feats, labels = batch_data[0], batch_data[1]
                else:
                    continue
                feats = feats.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = model(feats)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

        return time.time() - t0

    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


@torch.no_grad()
def evaluate_tabular_model(
    model: Any,
    dataset: Dataset,
    test_idx: List[int],
    device: torch.device = None,
    batch_size: int = 256,
) -> Tuple[float, float]:
    """
    统一评估接口。

    Returns:
        (accuracy, macro_f1)
    """
    from sklearn.metrics import accuracy_score, f1_score

    if isinstance(model, (XGBoostTabularWrapper, LightGBMWrapper)):
        X_test, y_test = extract_numpy_from_dataset(dataset, test_idx)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        return acc, f1

    elif isinstance(model, TabularMLP):
        model.eval()
        subset = torch.utils.data.Subset(dataset, test_idx)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

        all_preds, all_labels = [], []
        for batch_data in loader:
            if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 2:
                feats, labels = batch_data[0], batch_data[1]
            else:
                continue
            feats = feats.to(device)
            logits = model(feats)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        return acc, f1

    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


@torch.no_grad()
def get_tabular_probs_and_features(
    model: Any,
    dataset: Dataset,
    indices: List[int],
    device: torch.device = None,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    统一概率和特征提取接口（用于 AL 查询）。

    Returns:
        (probs, features)
    """
    if isinstance(model, (XGBoostTabularWrapper, LightGBMWrapper)):
        X, _ = extract_numpy_from_dataset(dataset, indices)
        probs = model.predict_proba(X)
        features = model.get_features(X)
        return probs, features

    elif isinstance(model, TabularMLP):
        model.eval()
        subset = torch.utils.data.Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

        all_probs, all_features = [], []
        for batch_data in loader:
            if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 2:
                feats = batch_data[0]
            else:
                feats = batch_data
            feats = feats.to(device)
            logits, features = model(feats, return_features=True)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_features.append(features.cpu().numpy())

        return np.vstack(all_probs), np.vstack(all_features)

    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
