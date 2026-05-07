import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huấn luyện mô hình và ghi nhận kết quả vào MLflow.

    Tham số:
        params: dict chứa các siêu tham số cho RandomForestClassifier
        data_path: đường dẫn đến file dữ liệu huấn luyện
        eval_path: đường dẫn đến file dữ liệu đánh giá

    Trả về:
        accuracy (float): độ chính xác trên tập đánh giá
    """

    # 1.5.1: Đọc dữ liệu huấn luyện và đánh giá
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # 1.5.2: Tách đặc trưng và nhãn
    X_train = df_train.drop("target", axis=1)
    y_train = df_train["target"]
    X_eval = df_eval.drop("target", axis=1)
    y_eval = df_eval["target"]

    # 1.5.3: Bắt đầu một MLflow run
    with mlflow.start_run():
        # 1.5.4: Ghi nhận các siêu tham số vào MLflow
        mlflow.log_params(params)

        # 1.5.5: Khởi tạo và huấn luyện mô hình
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # 1.5.6: Tính accuracy và f1_score
        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        # 1.5.7: Ghi nhận các chỉ số vào MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # 1.5.8: Log mô hình vào MLflow artifact
        mlflow.sklearn.log_model(model, "model")

        # 1.5.9: In kết quả ra màn hình
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # 1.5.10: Lưu metrics ra file outputs/metrics.json
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f)

        # 1.5.11: Lưu mô hình ra file models/model.pkl
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # 1.5.12: Trả về acc
    return acc


if __name__ == "__main__":
    # Đọc siêu tham số từ params.yaml và gọi hàm train()
    if not os.path.exists("params.yaml"):
        print("params.yaml not found. Creating one with default values.")
        with open("params.yaml", "w") as f:
            yaml.dump({"n_estimators": 100, "max_depth": 5, "min_samples_split": 2}, f)

    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    
    # Thiết lập MLflow tracking URI nếu chưa được set trong env
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
    train(params)
