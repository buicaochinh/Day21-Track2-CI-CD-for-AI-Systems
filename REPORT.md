# Báo Cáo Thực Hành Lab MLOps: Wine Quality Pipeline

**Học viên:** [Tên của bạn]
**Ngày thực hiện:** 07/05/2026

---

## 1. Kết quả Bước 1: Thực nghiệm cục bộ với MLflow

### Bộ siêu tham số tốt nhất
Dựa trên các thí nghiệm được theo dõi qua MLflow UI, bộ siêu tham số sau đây đã được chọn:

- `n_estimators`: 1000
- `max_depth`: 30
- `min_samples_split`: 2

**Lý do chọn:** Bộ tham số này cho kết quả Accuracy cao nhất trên tập `eval.csv` trong các lần chạy thử nghiệm cục bộ, giúp mô hình học được các đặc trưng phức tạp của dữ liệu rượu vang.

---

## 2. Kết quả Bước 2 & 3: Pipeline CI/CD và Huấn luyện liên tục

Hệ thống đã tự động hóa quy trình từ Code -> DVC Data -> GitHub Actions -> Deploy lên Cloud VM.

### Bảng So Sánh Hiệu Suất

| Chỉ số | Bước 2 (2998 mẫu) | Bước 3 (5996 mẫu) | Nhận xét |
|---|---|---|---|
| **Accuracy** | 0.678 | 0.76 | Tăng đáng kể nhờ nạp thêm dữ liệu mới. |
| **F1-Score** | 0.676 | 0.7593 | Độ tin cậy của mô hình cải thiện đồng nhất. |

*(Lưu ý: Bạn hãy lấy số Accuracy chính xác từ log của GitHub Actions để điền vào dấu ?)*

---

## 3. Nhật ký triển khai và Khó khăn gặp phải

### Khó khăn và Cách giải quyết:
1.  **Lỗi Push Git do file MLflow quá lớn:** Ban đầu tôi lỡ commit thư mục `mlruns/` chứa các file `.pkl` nặng hơn 100MB.
    *   *Giải quyết:* Cập nhật `.gitignore` và dùng `git rm --cached` để gỡ bỏ các file rác khỏi Git index.
2.  **Lỗi Permission Denied trong Unit Test:** GitHub Actions runner (Linux) bị lỗi khi truy cập đường dẫn `/Users` của máy Mac.
    *   *Giải quyết:* Chỉnh sửa code test để ép buộc MLflow sử dụng đường dẫn tương đối và thiết lập biến môi trường trước khi khởi động.
3.  **Lỗi thiếu Checkout trong Job Deploy:** Bước upload file API bị lỗi vì runner mới chưa có source code.
    *   *Giải quyết:* Bổ sung `actions/checkout@v4` vào Job Deploy.

---

## 4. Xác nhận hệ thống hoạt động
- **GitHub Actions:** Tất cả các job (Test, Train, Eval, Deploy) đều xanh ở Bước 3.
- **Inference API:**
    - URL: `http://136.116.98.86:8000/health` -> Trả về `{"status": "ok"}`
    - URL: `http://136.116.98.86:8000/predict` -> Trả về kết quả dự đoán đúng định dạng.

---
*Báo cáo được thực hiện với sự hỗ trợ của Antigravity AI.*
