# MDD-Challenge-2026

## 📂 Cấu trúc Thư mục

```text
MDD-Challenge-2026/
├── config.yaml                # Tệp cấu hình duy nhất (Hyperparameters, Paths, Flags)
│
├── data/                      # (Git-ignored) Thư mục quản lý dữ liệu
│   ├── raw/                   # Dữ liệu gốc (Audio .wav, Metadata CSV)
│   └── processed/             # Dữ liệu đã tiền xử lý (Splits, vocab.json)
│
├── src/                       # Mã nguồn chính của dự án
│   ├── data/                  # Phân hệ Dữ liệu
│   │   ├── __init__.py
│   │   ├── dataset.py         # Xử lý logic tải Audio & Text (PhonemeDataset)
│   │   └── collator.py        # Logic ghép Batch và đệm "-100" (CTC Padding)
│   │
│   ├── models/                # Phân hệ Kiến trúc AI
│   │   ├── __init__.py
│   │   └── builder.py         # Khởi tạo mô hình Wav2Vec2ForCTC và Processor
│   │
│   ├── training/              # Phân hệ Huấn luyện
│   │   ├── __init__.py
│   │   ├── train.py           # File chạy chính (Khởi chạy Hugging Face Trainer)
│   │   └── metrics.py         # Hàm đánh giá (WER, PER)
│   │
│   ├── inference/             # Phân hệ Thực chiến (Suy luận)
│   │   ├── __init__.py
│   │   └── predict.py         # Pipeline load model và dự đoán đầu ra IPA
│   │
│   └── utils/                 # Phân hệ Công cụ hỗ trợ
│       ├── __init__.py
│       ├── audio_utils.py     # Hàm xử lý âm thanh (Resample 16kHz, Stereo to Mono)
│       ├── evaluate.py        # Script tính điểm chuẩn (WER/PER) do BTC cung cấp
│       └── file_utils.py      # Hàm tương tác I/O (Đọc YAML, JSON
│
├── experiments/               # (Git-ignored) Nơi tự động lưu Checkpoints và Logs
├── notebooks/                 # Jupyter notebooks để phân tích EDA và Error Analysis
├── requirements.txt           # Danh sách các thư viện Python cần thiết
├── .gitignore                
└── README.md                  