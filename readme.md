🚗 ODOMETER READING SYSTEM

An end-to-end AI-powered odometer reading system that automatically classifies odometer type, detects the odometer region, and extracts numeric readings from vehicle images using Deep Learning, YOLO object detection, and OCR fallback methods.


This project includes:

-A Flask web application for image upload and visualization

-A multi-stage inference pipeline

-YOLO-based training scripts for odometer and digit detection

-Robust fallback mechanisms (OCR + heuristics)



📌 OVERVIEW

This system processes an uploaded image and performs the following steps:

-Odometer Classification

Classifies the odometer as Analog or Digital using a CNN (ResNet-based).

-Odometer Region Detection

Locates the odometer in the image using YOLO object detection.

-Digit Detection & Reading Extraction

Primary: YOLO-based digit detection with advanced filtering

-Fallback: Tesseract OCR

Final fallback: heuristic-based estimation

Visualization & Confidence Reporting

Bounding boxes, detected digits, confidence scores, and processing metadata are rendered.


🧠 SYSTEM ARTITECTURE
```
Image Upload

     │
     
     ▼
     
Classification (ResNet)

     │
     ▼
Odometer Detection (YOLO)

     │
     ▼
Digit Detection (YOLO)

     │
     ├─ Improved Filtering
     ├─ OCR (Tesseract)
     └─ Heuristic Fallback

     ▼
Final Reading + Visualization
```


✨ FEATURES


-Flask-based web UI & REST API

-GPU/CPU auto-detection (PyTorch)

-YOLO-based odometer & digit detection

-Advanced digit filtering:

-Vertical alignment

-Height clustering

-Horizontal grouping

-OCR fallback using Tesseract

-Model health & status endpoints

-Dashboard for uploaded images

-Training, tuning, and evaluation scripts

📁 PROJECT STRUCTURE
```
project_root/
│
├── web_app/
│   └── app.py                # Flask application & inference pipeline
│
├── detection/
│   └── train_detection.py    # YOLO training & evaluation CLI
│
├── dataset/                  # Odometer detection dataset
├── dataset_digits/           # Digit detection dataset (auto-labeled)
│
├── models/
│   └── classification/       # Trained CNN classification model
│
├── runs/train/               # YOLO training outputs
├── uploads/                  # Uploaded & processed images
├── config.yaml               # Training configuration (optional)
└── README.md
```

⚙️ INSTALLATION

1. Clone the repository

```
git clone https://github.com/Meliodus254/Odometer-Mileage-Reader.git
cd project_root
```

2. Create a virtual environment (recommended)

```
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```


3. Install dependencies

```
pip install -r requirements.txt
```

Note: ultralytics will be auto-installed by the training script if missing.


▶️ Running the Web App

```
python web_app/app.py
```

The server starts at:

```

http://localhost:5000
```



🏋️ MODEL TRAINING

Run the interactive training menu:

```

python detection/train_detection.py
```

Training Options

-Train odometer detector (Stage 1)

-Train digit detector (Stage 2)

-Train both stages

-Evaluate odometer model

-Evaluate digit model

-Auto-label digit dataset



🔧 CONFIGURATION

Optional training config via config.yaml:


📊 EVALUATION

After training, models can be evaluated automatically or manually.

Metrics include:

-Precision

-Recall

-mAP@0.5

-mAP@0.5:0.95

Evaluation results can be exported as JSON.



🚀 Future Improvements


-Temporal odometer validation (reading consistency)

-Video-based odometer tracking

-Mobile-friendly UI

-Cloud deployment (Docker / AWS)

-Support for analog needle-based readings


📄 License


This project is licensed under the MIT License.
Feel free to use, modify, and distribute.

