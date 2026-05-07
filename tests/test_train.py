import os
import json
import numpy as np
import pandas as pd
import pytest
import mlflow
from src.train import train

# Thiết lập MLflow tracking cho môi trường test
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow_test.db"
os.environ["MLFLOW_ARTIFACT_ROOT"] = "./mlartifacts_test"

FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]

def _make_temp_data(tmp_path):
    """
    Tạo dataset nhỏ với cùng schema Wine Quality để sử dụng trong test.
    """
    rng = np.random.default_rng(0)
    n = 200
    
    # 2.10.1: Tạo mảng X ngẫu nhiên
    X = rng.random((n, len(FEATURE_NAMES)))
    
    # 2.10.2: Tạo mảng y ngẫu nhiên (0, 1, 2)
    y = rng.integers(0, 3, size=n)
    
    # 2.10.3: Tạo DataFrame
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y
    
    # 2.10.4: Lưu vào file csv tạm thời
    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "eval.csv"
    
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)
    
    # 2.10.5: Trả về đường dẫn
    return str(train_path), str(eval_path)


def test_train_returns_float(tmp_path):
    """Kiểm tra hàm train() trả về một số thực trong khoảng [0, 1]."""
    train_path, eval_path = _make_temp_data(tmp_path)
    
    # 2.10.6: Gọi hàm train với params nhỏ
    acc = train(
        {"n_estimators": 10, "max_depth": 3, "min_samples_split": 2},
        data_path=train_path,
        eval_path=eval_path
    )
    
    # 2.10.7: Assert kết quả
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_metrics_file_created(tmp_path):
    """Kiểm tra file outputs/metrics.json được tạo sau khi huấn luyện."""
    train_path, eval_path = _make_temp_data(tmp_path)
    
    # Chạy train
    train(
        {"n_estimators": 10, "max_depth": 3, "min_samples_split": 2},
        data_path=train_path,
        eval_path=eval_path,
    )
    
    # 2.10.8: Assert file metrics.json tồn tại
    assert os.path.exists("outputs/metrics.json")
    
    # 2.10.9: Kiểm tra nội dung
    with open("outputs/metrics.json", "r") as f:
        metrics = json.load(f)
        assert "accuracy" in metrics
        assert "f1_score" in metrics


def test_model_file_created(tmp_path):
    """Kiểm tra file models/model.pkl được tạo sau khi huấn luyện."""
    train_path, eval_path = _make_temp_data(tmp_path)
    
    # Chạy train
    train(
        {"n_estimators": 10, "max_depth": 3, "min_samples_split": 2},
        data_path=train_path,
        eval_path=eval_path,
    )
    
    # 2.10.10: Assert file model.pkl tồn tại
    assert os.path.exists("models/model.pkl")
