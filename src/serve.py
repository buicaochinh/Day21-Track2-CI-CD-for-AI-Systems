from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os
import uvicorn

app = FastAPI()

# Đọc tên bucket từ biến môi trường
GCS_BUCKET = os.environ.get("GCS_BUCKET", "your-bucket-name")
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")

def download_model():
    """Tải file model.pkl từ GCS về máy khi server khởi động."""
    if not os.path.exists(os.path.expanduser("~/models")):
        os.makedirs(os.path.expanduser("~/models"))
        
    print(f"Downloading model from gs://{GCS_BUCKET}/{GCS_MODEL_KEY}...")
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_MODEL_KEY)
        blob.download_to_filename(MODEL_PATH)
        print(f"Model downloaded successfully to {MODEL_PATH}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        # Nếu không tải được (ví dụ lần đầu chưa có model), chúng ta vẫn tiếp tục 
        # nhưng model load sẽ lỗi sau đó.

# Tải model khi khởi động (chỉ chạy khi deploy thực tế trên VM)
if os.environ.get("DEPLOYMENT_ENV") == "production":
    download_model()

# Load model nếu file tồn tại
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print("Warning: Model file not found. API will return errors for /predict.")

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    """Endpoint kiểm tra sức khỏe server."""
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luận.
    Đầu vào: JSON {"features": [f1, f2, ..., f12]}
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    # Dự đoán
    prediction = int(model.predict([req.features])[0])
    
    # Nhãn: 0 -> "thấp", 1 -> "trung bình", 2 -> "cao"
    labels = {0: "thấp", 1: "trung bình", 2: "cao"}
    
    return {
        "prediction": prediction,
        "label": labels.get(prediction, "unknown")
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
