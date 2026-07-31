# SMARTTRAFFIC - AI: Hệ Thống Giám Sát Giao Thông Thông Minh

**SMARTTRAFFIC - AI** là ứng dụng giám sát và phân tích giao thông sử dụng công nghệ Thị giác máy tính (Computer Vision) và Trí tuệ nhân tạo (AI). Hệ thống cung cấp khả năng nhận diện phương tiện trong video, theo dõi liên tục qua nhiều frame, tự động tính toán mật độ giao thông trong vùng quan tâm (ROI), đồng thời phát hiện vi phạm vượt đèn đỏ và lưu trữ ảnh bằng chứng trực quan.

---

## 📌 Mục Tiêu Dự Án

* **Nhận diện & Đếm phương tiện:** Sử dụng mô hình YOLOv8 để phát hiện các lớp phương tiện giao thông (như `car`, `motorcycle`, `bus`, `truck`, `container_truck`, `van`, `bicycle`, `fire_engine`).
* **Theo dõi đa đối tượng (Multi-Object Tracking):** Tích hợp thuật toán **ByteTrack** gán `track_id` duy nhất cho từng phương tiện, giúp đếm chính xác và tránh ghi trùng lặp vi phạm.
* **Đánh giá Mật độ Giao thông (ROI Density):** Tính toán mật độ xe trong vùng ROI theo thời gian thực và phân loại trạng thái: `Bình thường`, `Đông`, hoặc `Ùn tắc`.
* **Phát hiện Vi phạm Đèn đỏ:** Khi tín hiệu đèn là `RED`, tự động phát hiện phương tiện cắt qua vạch ảo theo hướng quy định, lưu thông tin vi phạm vào CSDL SQLite và xuất ảnh bằng chứng có khung khoanh vùng (bounding box).
* **Dashboard Trực Quan:** Giao diện Web hiện đại, hỗ trợ tải video, tùy chỉnh tham số, stream khung hình kèm bounding box và xem báo cáo tổng kết phiên xử lý.

---

## 🏗️ Kiến Trúc Hệ Thống

```text
Video Upload 
  │
  ▼
FastAPI Session Controller
  │
  ├─► OpenCV Frame Capture
  │     │
  │     ▼
  ├─► YOLOv8 Detection + ByteTrack Tracking
  │     │
  │     ▼
  ├─► ROI Density Calculation & Red-Light Crossing Logic
  │     │
  │     ▼
  ├─► Frame Annotation + Evidence Snapshot Generation
  │     │
  │     ▼
  └─► SQLite Violation Database (`violations.sqlite3`)
  │
  ▼
HTML5 / CSS3 / Vanilla JS Web Dashboard (HTTP / Stream)
```

### 🛠️ Công Nghệ Sử Dụng

* **Frontend:** HTML5, CSS3, Vanilla JavaScript (Fetch API, SSE / Polling).
* **Backend:** Python 3.10+, FastAPI, Uvicorn, Pydantic.
* **Computer Vision & AI:** OpenCV, Ultralytics YOLOv8 (PyTorch), ByteTrack.
* **Database & Storage:** SQLite3, Local File System (Lưu ảnh bằng chứng).
* **Testing:** Pytest (Unit test & Integration test).

---

## 📂 Cấu Trúc Thư Mục Dự Án

```text
project_final/
├── backend/                  # Mã nguồn Backend FastAPI
│   ├── main.py               # API Endpoints & Cấu hình Routes
│   ├── models.py             # Schema Pydantic cho Request/Response
│   ├── storage.py            # Quản lý CSDL SQLite & Lưu trữ Ảnh bằng chứng
│   └── tracker.py            # Tích hợp ByteTrack & YOLOv8 Processing
├── configs/
│   └── config.yaml           # Cấu hình mặc định (ROI, Vạch vi phạm, Thresh)
├── core/                     # Các module xử lý lõi (Core Logic)
│   ├── density.py            # Tính toán mật độ giao thông vùng ROI
│   ├── detector.py           # Nhận diện đối tượng với YOLOv8
│   ├── violation.py          # Logic kiểm tra xe cắt vạch khi đèn đỏ
│   └── runtime.py            # Cấu hình môi trường thực thi
├── data/                     # Dữ liệu & Tập huấn luyện (Dataset)
│   ├── sample_videos/        # Video mẫu test ứng dụng
│   └── vehicle_dataset/      # Dataset YOLO (data.yaml, train, valid, test)
├── evidence/                 # Thư mục chứa ảnh chụp bằng chứng vi phạm
├── models/                   # Chứa các file trọng số (.pt) của YOLOv8
│   └── vehicle_best.pt       # Mô hình đã được train custom
├── tools/                    # Các công cụ hỗ trợ
│   └── train_vehicle_model.py# Script huấn luyện mô hình tùy chỉnh (GPU/CPU)
├── tests/                    # Thư mục kiểm thử Pytest
├── run.ps1                   # Script khởi động dự án trên Windows PowerShell
├── run.bat                   # Script khởi động dự án trên Command Prompt
└── requirements.txt          # Danh sách thư viện phụ thuộc
```

---

## 💻 Hướng Dẫn Cài Đặt & Chạy Ứng Dụng

### 1. Yêu Cầu Môi Trường
* **Hệ điều hành:** Windows 10/11 (hoặc Linux/macOS).
* **Python:** Phiên bản 3.10, 3.11 hoặc 3.12.

### 2. Cài Đặt Thư Viện

Mở PowerShell tại thư mục gốc của dự án:

```powershell
# Tạo môi trường ảo (nếu chưa có)
python -m venv .venv

# Kích hoạt môi trường ảo
.\.venv\Scripts\activate

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# Nâng cấp typing-extensions để đảm bảo tương thích với FastAPI & Pydantic
.\.venv\Scripts\python.exe -m pip install -U typing-extensions
```

*(Tùy chọn) Cài đặt PyTorch hỗ trợ GPU NVIDIA CUDA (ví dụ cho RTX 3050, RTX 4060...):*
```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

### 3. Khởi Động Web Dashboard

Khởi động hệ thống nhanh bằng lệnh:

```powershell
.\run.ps1
```
*(Hoặc chạy lệnh uvicorn trực tiếp):*
```powershell
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Sau khi khởi động thành công, mở trình duyệt web và truy cập:
👉 **`http://127.0.0.1:8000`**

---

## 🎯 Huấn Luyện Mô Hình Tùy Chỉnh (Custom Training)

Dự án đi kèm script `tools/train_vehicle_model.py` hỗ trợ huấn luyện lại YOLOv8 với tập dữ liệu riêng (như tập dữ liệu trong `data/vehicle_dataset/data.yaml`).

### 1. Cấu hình Tập Dữ Liệu (`data/vehicle_dataset/data.yaml`)
Cấu hình đường dẫn và các lớp phương tiện:
```yaml
path: data/vehicle_dataset
train: train/images
val: valid/images
test: test/images

nc: 8
names: ['bicycle', 'bus', 'car', 'container_truck', 'fire_engine', 'motorcycle', 'truck', 'van']
```

### 2. Lệnh Huấn Luyện Bằng GPU NVIDIA (Khuyên dùng)
```powershell
python tools/train_vehicle_model.py --data data/vehicle_dataset/data.yaml --base-model yolov8s.pt --device 0 --epochs 60 --imgsz 640 --batch 16
```

### 3. Lệnh Huấn Luyện Bằng CPU
```powershell
python tools/train_vehicle_model.py --data data/vehicle_dataset/data.yaml --base-model yolov8s.pt --device cpu --epochs 30 --imgsz 640 --batch 8
```

> **Sau khi huấn luyện hoàn tất:** Mô hình có độ chính xác cao nhất (`best.pt`) sẽ tự động được sao chép vào `models/vehicle_best.pt`. Bạn chỉ cần chọn `models/vehicle_best.pt` trên giao diện Dashboard để sử dụng.

---

## ⚙️ Cấu Hình ROI & Vi Phạm (`configs/config.yaml`)

Bạn có thể chỉnh sửa file `configs/config.yaml` để điều chỉnh thông số hoạt động:

```yaml
model_path: yolov8n.pt
confidence_threshold: 0.35

# Tối đa số phương tiện dự kiến trong vùng ROI để tính % mật độ
max_capacity: 30

# Ngưỡng mật độ (%): 0-40 (Bình thường), 41-70 (Đông), >70 (Ùn tắc)
density_threshold:
  normal: 40
  crowded: 70

# Vùng quan tâm ROI (tỉ lệ 0.0 -> 1.0 theo x1, y1, x2, y2)
roi_ratio:
  x1: 0.0
  y1: 0.0
  x2: 1.0
  y2: 1.0

# Vạch ảo phát hiện vượt đèn đỏ (tỉ lệ chiều cao khung hình 0.0 -> 1.0)
line_position_ratio: 0.62

# Hướng di chuyển xe cắt vạch tính vi phạm: down, up, hoặc both
line_crossing_direction: down
```

---

## 📡 Danh Sách API Endpoints

| HTTP Method | API Endpoint | Mô tả |
| :--- | :--- | :--- |
| `GET` | `/` | Trả về giao diện Dashboard Web |
| `GET` | `/api/health` | Kiểm tra trạng thái hoạt động của server |
| `GET` | `/api/models` | Trả về danh sách các model có sẵn trong thư mục `models/` |
| `POST` | `/api/sessions` | Khởi tạo phiên xử lý video mới (upload video & nhận config) |
| `POST` | `/api/sessions/{session_id}/next-frame` | Xử lý frame tiếp theo của video |
| `GET` | `/api/sessions/{session_id}/summary` | Lấy tổng kết chỉ số của phiên làm việc hiện tại |
| `DELETE`| `/api/sessions/{session_id}` | Dừng và dọn dẹp phiên xử lý |
| `GET` | `/api/violations` | Đọc danh sách log vi phạm từ CSDL SQLite |
| `GET` | `/api/evidence/{relative_path}` | Tải/Xem ảnh bằng chứng vi phạm |

---

## 🧪 Kiểm Thử (Testing)

Hệ thống tích hợp bộ kiểm thử tự động với Pytest. Để chạy kiểm thử:

```powershell
pip install -r requirements-dev.txt
python -m pytest
```

Bộ test đảm bảo kiểm tra toàn bộ các thành phần:
* Logic tính toán mật độ giao thông vùng ROI.
* Logic phát hiện xe cắt qua vạch vi phạm đèn đỏ.
* Đọc/ghi CSDL vi phạm SQLite.
* Kiểm tra tính hợp lệ của Model Registry và API Endpoints.

---

## 📖 Cơ Sở Lý Thuyết & So Sánh Phương Pháp

* **YOLOv8 (You Only Look Once v8):** Thuật toán phát hiện đối tượng Single-stage State-of-the-Art, cho tốc độ xử lý nhanh, độ chính xác cao phù hợp với ứng dụng thời gian thực (Real-time).
* **ByteTrack:** Thuật toán tracking vượt trội hơn DeepSORT nhờ việc tận dụng cả các Bounding Box có độ tin cậy thấp (low-score detection boxes), giúp hạn chế tối đa tình trạng mất dấu đối tượng khi xe bị che khuất tạm thời.
* **So với CNN phân loại thuần túy:** Phân loại CNN truyền thống chỉ phân loại toàn bộ khung hình, không thể xác định vị trí (Bounding Box), không đếm được số lượng từng loại xe và không có khả năng theo dõi vết di chuyển (Tracking).

---

## 📝 Giấy Phép & Tác Giả

* **Dự án:** SMARTTRAFFIC - AI (Báo cáo Đồ án Thị giác Máy tính).
* **Phát triển bởi:** Đội ngũ phát triển SMARTTRAFFIC.
* **Giấy phép:** Được bảo hộ và phát triển cho mục đích học thuật & nghiên cứu.
