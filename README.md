# Handwritten Text Detection with YOLOv8 & Custom CNN

This project implements a **two-stage pipeline** for handwritten document analysis:
1. **YOLOv8 (YOLO26n)** - Object detection for `check` and `area` localization
2. **Custom CNN** - Binary classification for check verification

---

## 📁 Project Structure

```
Handwritten/
├── data_config.yaml              # YOLO dataset configuration
├── yolo_detection.ipynb          # Main YOLO training & evaluation (full)
├── labelme_to_yolo_format.ipynb  # LabelMe JSON → YOLO format converter
├── LabelmeToYolo.ipynb           # Duplicate of above (alternative name)
├── YoloRun.ipynb                 # Minimal YOLO train/inference script
├── CNN.ipynb                     # Custom CNN binary classifier
├── DataPruning.ipynb             # YOLO-guided cropping for CNN data prep
├── generate_names.py             # Synthetic Persian check generator
├── inference.py                  # Production inference script (YOLO)
├── data_config.yaml              # YOLO dataset config
├── logs.txt                      # CNN training logs (50 epochs)
├── LICENSE                       # MIT License
├── test.jpg                      # Test image for inference
├── yolo26n.pt                    # Pre-trained YOLOv8n weights
├── weights/
│   └── yolo26n.pt               # Backup weights
├── best_model.pth                # Best CNN model (binary classification)
├── checkpoint.pth                # Full CNN checkpoint (epoch+optimizer)
├── CNN_Acc.png                   # CNN accuracy curves
├── CNN_Loss.png                  # CNN loss curves
├── CNN_CM.png                    # CNN confusion matrix
├── yolo_dataset/                 # YOLO detection dataset
│   ├── images/
│   │   ├── train/ (321 images)
│   │   └── val/ (89 images)
│   └── labels/
│       ├── train/ (YOLO format)
│       ├── val/ (YOLO format)
│       ├── train.cache
│       └── val.cache
├── Dataset/                      # CNN classification dataset
│   ├── images/                   # All images
│   ├── labels/                   # Binary labels (0/1)
│   ├── Positives/                # Class 1 (check present)
│   └── Negatives/                # Class 0 (no check)
└── runs/
    └── detect/
        └── train/                # YOLO training outputs
            ├── weights/
            │   ├── best.pt       # Best YOLO model (mAP50-95)
            │   └── last.pt       # Last epoch YOLO model
            ├── results.png       # Training metrics
            ├── confusion_matrix.png
            └── ...
```

---

## 🔧 Requirements

```bash
pip install ultralytics opencv-python torch torchvision torchinfo albumentations scikit-learn matplotlib pillow
```

- Python 3.10+
- PyTorch 2.5+ with CUDA
- Ultralytics 8.4+
- OpenCV, Albumentations, torchinfo
- NVIDIA GPU (RTX 3060 Laptop used)

---

## 📊 Datasets

### 1. YOLO Detection Dataset (`yolo_dataset/`)
| Split | Images | Classes |
|-------|--------|---------|
| Train | 321 | check (0), area (1) |
| Val | 89 | check (0), area (1) |

- Images: 640×640 grayscale
- Labels: YOLO format (normalized xywh)

### 2. CNN Classification Dataset (`Dataset/`)
| Class | Label | Description | Count |
|-------|-------|-------------|-------|
| Positive | 1 | Check present | ~50% |
| Negative | 0 | No check | ~50% |

- Images: 64×256 grayscale (resized from YOLO crops)
- Labels: Single integer per file (0 or 1)

---

## 📓 Notebooks Documentation

### 1. `labelme_to_yolo_format.ipynb` / `LabelmeToYolo.ipynb`
**LabelMe JSON → YOLO format converter + image preprocessing**

**Workflow:**
1. Parse LabelMe JSON (rectangle annotations with 2 corner points)
2. Convert to YOLO format: `class_id x_center y_center width height` (normalized)
3. Resize images to 640×640 grayscale

**Key Assumptions:**
- Exactly 2 shapes per image: `shapes[0]` = check, `shapes[1]` = area
- Rectangle annotations (2 points per shape)

### 2. `yolo_detection.ipynb`
**Full YOLOv8 training, validation, and analysis**

**Model: YOLO26n (YOLOv8n variant)**
- Parameters: 2.5M | GFLOPs: 5.9 | Layers: 260
- Backbone: CSPDarknet with C3k2 blocks
- Neck: PAN-FPN with C2PSA attention
- Head: Detect (3 scales: 64, 128, 256)

**Training Config (50 epochs):**
| Param | Value |
|-------|-------|
| Batch | 16 |
| Img Size | 640 |
| Optimizer | AdamW (auto) |
| LR | 0.001667 |
| Weight Decay | 0.0005 |
| Augmentation | Mosaic=1.0, HSV, FlipLR=0.5, Erasing=0.4 |
| AMP | Enabled |
| Patience | 100 |

**Results (Epoch 50):**
| Metric | Value |
|--------|-------|
| mAP50 (all) | 0.995 |
| mAP50-95 (all) | 0.846 |
| Box Loss | 0.580 |
| Cls Loss | 0.331 |

**Per-Class:**
| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| check (0) | 0.991 | 1.000 | 0.995 | 0.955 |
| area (1) | 0.997 | 1.000 | 0.995 | 0.736 |

**Speed:** ~3.8ms/img (preprocess 0.6ms + inference 1.6ms + postprocess 1.6ms)

### 3. `YoloRun.ipynb`
**Minimal YOLO script (2 cells):**
```python
# Cell 1: Train
model = YOLO("yolo26n.pt")
model.train(data="data_config.yaml", epochs=50, imgsz=640, batch=16)

# Cell 2: Inference
model = YOLO("runs/detect/train/weights/best.pt")
results = model("test.jpg")
results[0].show()
```

### 4. `CNN.ipynb`
**Custom CNN for binary check classification**

**Architecture:**
```
Input: 1×64×256 (grayscale)
├─ Conv2d(1→64, 3×3) + Conv2d(64→128, 3×3) + MaxPool2d(2) + BN + ReLU
├─ Conv2d(128→256, 3×3) + MaxPool2d(2) + BN + ReLU
├─ Conv2d(256→64, 3×3) + MaxPool2d(2) + BN + ReLU
├─ Conv2d(64→16, 3×3) + MaxPool2d(2) + ReLU
├─ Flatten → Dropout(0.2) → Linear(1024→16) + ReLU
└─ Linear(16→1) + Sigmoid
```
- Parameters: 543,729 | Mult-Adds: 41.33G

**Training (50 epochs):**
- Loss: BCELoss
- Optimizer: Adam (lr=1e-3, weight_decay=1e-2)
- Batch: 16 | Split: 80/20 train/test

**Final Metrics (Epoch 50):**
| Metric | Train | Test |
|--------|-------|------|
| Loss | 0.1725 | 0.2382 |
| Accuracy | 0.9357 | 0.9013 |

**Best Test Accuracy:** 0.9454 (Epoch 41)
**Best Test Loss:** 0.1595 (Epoch 41)

**Outputs:**
- `best_model.pth` - Best state_dict only
- `checkpoint.pth` - Full checkpoint (epoch, model, optimizer, loss)
- `CNN_Acc.png`, `CNN_Loss.png`, `CNN_CM.png` - Visualizations

### 5. `DataPruning.ipynb`
**YOLO-guided region cropping for CNN dataset preparation**

**Pipeline:**
1. Load trained YOLO (`best.pt`)
2. Iterate through `Dataset/Positives/` and `Dataset/Negatives/`
3. Run YOLO inference → get `area` class bounding box (index 1)
4. Crop detected region → resize to 256×64
5. Save to CNN training directories

**Purpose:** Extract only the relevant "area" region from full documents for efficient CNN classification.

### 6. `generate_names.py`
**Synthetic Persian check image generator**

**Features:**
- Base template: `in_payment_of.png`
- 22 Persian names (companies + persons)
- Random Persian fonts from `fonts/` directory
- RTL text rendering with RAQM/arabic_reshaper+bidi fallback
- Random ink intensity (5-20), rotation (-0.8° to 0.8°)
- High-res rendering (6× scale) → downsample with LANCZOS
- Output: 500 PNG files + ZIP archive
- Filenames: UUID-based

**Config:**
```python
TOTAL_COUNT = 500
BOX_X, BOX_Y = 420, 130      # Text box position
BOX_WIDTH, BOX_HEIGHT = 400, 150
SCALE_FACTOR = 6              # Supersampling for quality
```

---

## 🚀 Usage

### YOLO Training
```bash
# Full notebook
jupyter notebook yolo_detection.ipynb

# Minimal
jupyter notebook YoloRun.ipynb

# CLI
yolo detect train data=data_config.yaml model=yolo26n.pt epochs=50 imgsz=640 batch=16 device=0
```

### YOLO Inference
```python
from ultralytics import YOLO
model = YOLO("runs/detect/train/weights/best.pt")
results = model.predict("test.jpg", imgsz=640, conf=0.25)
# results[0].boxes.xywhn, .cls, .conf
```

### CNN Training
```bash
jupyter notebook CNN.ipynb
```

### CNN Inference
```python
import torch, cv2
model = CNN()  # Define architecture
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 64)) / 255.0
img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    prob = model(img).item()  # 0.0-1.0, >0.5 = check present
```

### Full Pipeline (YOLO + CNN)
```python
# 1. YOLO detects check + area boxes
# 2. Crop area region using YOLO box
# 3. CNN classifies if check is present in cropped region
```

### Synthetic Data Generation
```bash
python generate_names.py
# Creates 500 images in generated_checks_huge/ + persian_filled_checks_huge.zip
```

### Production Inference Script
```bash
python inference.py --image test.jpg --model runs/detect/train/weights/best.pt
python inference.py --dir test_images/ --output results/ --conf 0.3
```

---

## 📈 Key Achievements

| Component | Metric | Value |
|-----------|--------|-------|
| **YOLO Detection** | mAP50 | 0.995 |
| | mAP50-95 | 0.846 |
| | Inference Speed | 3.8 ms/img |
| | Model Size | 5.4 MB |
| **CNN Classification** | Best Test Acc | 0.9454 |
| | Final Test Acc | 0.9013 |
| | Model Size | ~2.1 MB |
| **Combined Pipeline** | End-to-end | Detection + Verification |

---

## 🔄 Complete Pipeline Flow

```
┌─────────────────┐
│ LabelMe JSON    │
└────────┬────────┘
         │ labelme_to_yolo_format.ipynb
         ▼
┌─────────────────┐     ┌─────────────────┐
│ YOLO Dataset    │     │ Synthetic Data  │
│ (640×640 gray)  │     │ generate_names.py
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ YOLO Training   │     │ Augment Dataset │
│ (yolo_detection)│     │ (optional)      │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ best.pt         │────▶│ DataPruning     │
│ (YOLO detect)   │     │ (crop areas)    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ Detect check/   │     │ CNN Dataset     │
│ area boxes      │     │ (64×256 crops)  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
          ┌─────────────────┐
          │ CNN Training    │
          │ (CNN.ipynb)     │
          └────────┬────────┘
                   ▼
          ┌─────────────────┐
          │ best_model.pth  │
          │ (binary check   │
          │  classifier)    │
          └─────────────────┘
```

---

## 📝 Notes

1. **YOLO Input:** Single-channel grayscale (adapted from 3-channel pretrained weights)
2. **Transfer Learning:** 606/708 YOLO layers transferred from COCO pretrained
3. **CNN Input:** 64×256 crops from YOLO `area` detections
4. **Class Balance:** Both datasets approximately balanced (50/50)
5. **Early Stopping:** YOLO patience=100, CNN saves best by test loss
6. **LabelMe Assumption:** Fixed 2-shape order (check→area)

---

## 📄 License

MIT License - Copyright (c) 2026 Sina Eslami

---

*Documentation generated from all source code, notebooks, and training logs.*