# Handwritten Text Detection with YOLOv8

This project implements a handwritten text detection system using YOLOv8 (YOLO26n variant) to detect two classes: **check** (class 0) and **area** (class 1) in handwritten documents.

## 📁 Project Structure

```
Handwritten/
├── data_config.yaml              # Dataset configuration for YOLO training
├── yolo_detection.ipynb          # Main training and evaluation notebook
├── labelme_to_yolo_format.ipynb  # LabelMe to YOLO format conversion
├── yolo26n.pt                    # Pre-trained YOLOv8n model weights
├── weights/
│   └── yolo26n.pt               # Backup of model weights
├── yolo_dataset/                 # Dataset directory
│   ├── images/
│   │   ├── train/               # Training images (321 images)
│   │   └── val/                 # Validation images (89 images)
│   └── labels/
│       ├── train/               # Training labels (YOLO format)
│       ├── val/                 # Validation labels (YOLO format)
│       ├── train.cache          # Cached training labels
│       └── val.cache            # Cached validation labels
└── runs/
    └── detect/
        └── train/               # Training outputs (weights, logs, plots)
            ├── weights/
            │   ├── best.pt      # Best model weights
            │   └── last.pt      # Last epoch weights
            ├── labels.jpg       # Label visualization
            ├── results.png      # Training metrics plots
            └── ...
```

## 🔧 Requirements

```bash
pip install ultralytics opencv-python torch torchvision
```

- Python 3.10+
- PyTorch 2.5+ with CUDA support
- Ultralytics 8.4+
- OpenCV
- NVIDIA GPU (RTX 3060 Laptop GPU used in training)

## 📊 Dataset

The dataset consists of **410 images** split into:
- **Training**: 321 images
- **Validation**: 89 images

### Classes
| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | check | Checkbox/checkmark detection |
| 1 | area | Text area/region detection |

### Data Format
Images are preprocessed to **640×640 grayscale**. Labels are in YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized to [0, 1] relative to image dimensions.

## 📓 Notebooks Documentation

### 1. `labelme_to_yolo_format.ipynb`

Converts LabelMe JSON annotations to YOLO format text files.

#### Workflow:
1. **Load dependencies**: `os`, `shutil`, `json`, `cv2`
2. **Define paths**: Source images/labels, output directory
3. **Parse LabelMe JSON**: Extract polygon points for each shape
4. **Convert to YOLO format**:
   - Calculate bounding box from 2 corner points
   - Normalize coordinates by image width/height
   - Save as `.txt` files with class ID and normalized coordinates
5. **Resize images**: Convert to grayscale and resize to 640×640

#### Key Code Sections:

**Dependencies & Paths:**
```python
import os, shutil, json, cv2 as cv

images_path = "C:/Users/Sina/Desktop/HamrahTel/Handwritten/yolo_dataset/images/"
labels_path = "C:/Users/Sina/Desktop/HamrahTel/Handwritten/yolo_dataset/labels/"
save_txt_path = "C:/Users/Sina/Desktop/HamrahTel/Handwritten/yolo_dataset/text_labels/"
```

**LabelMe to YOLO Conversion:**
```python
# Assumes exactly 2 shapes per image: check (index 0) and area (index 1)
label_0 = data["shapes"][0]['label']  # check
label_1 = data["shapes"][1]['label']  # area

# Extract corner points (2 points per rectangle)
x0_0, y0_0 = data["shapes"][0]['points'][0]
x1_0, y1_0 = data["shapes"][0]['points'][1]

# Convert to YOLO format (normalized center-x, center-y, width, height)
x_center = ((x0 + x1) / 2) / img_width
y_center = ((y0 + y1) / 2) / img_height
width = abs(x1 - x0) / img_width
height = abs(y1 - y0) / img_height
```

**Image Preprocessing:**
```python
img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2GRAY)
new_img = cv.resize(img, (640, 640))
cv.imwrite(save_path, new_img)
```

#### Output:
- YOLO format label files in `yolo_dataset/text_labels/`
- Resized grayscale images in `yolo_dataset/images/`

---

### 2. `yolo_detection.ipynb`

Main training, validation, and inference notebook using Ultralytics YOLOv8.

#### Configuration (`data_config.yaml`):
```yaml
path: C:/Users/Sina/Desktop/HamrahTel/Handwritten/yolo_dataset/
train: images/train
val: images/val
names:
  0: check
  1: area
```

#### Model Architecture: YOLO26n (YOLOv8n variant)
- **Parameters**: 2,504,580 (2.5M)
- **GFLOPs**: 5.9
- **Layers**: 260
- **Input**: 640×640 grayscale (1 channel)
- **Output**: 2 classes (check, area)

**Architecture Summary:**
```
Backbone: CSPDarknet with C3k2 blocks
Neck: PAN-FPN with C2PSA attention
Head: Detect head with 3 scales [64, 128, 256]
```

#### Training Hyperparameters:
| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch Size | 16 |
| Image Size | 640 |
| Optimizer | AdamW (auto-selected) |
| Learning Rate | 0.001667 (auto) |
| Momentum | 0.9 |
| Weight Decay | 0.0005 |
| Warmup Epochs | 3 |
| Mosaic Augmentation | 1.0 |
| Mixup | 0.0 |
| Flip LR | 0.5 |
| HSV Augmentation | h=0.015, s=0.7, v=0.4 |
| Box Loss Weight | 7.5 |
| Class Loss Weight | 0.5 |
| DFL Loss Weight | 1.5 |
| Patience (Early Stop) | 100 |
| Workers | 8 |
| AMP | Enabled |
| Device | CUDA:0 |

#### Training Command:
```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # Load pre-trained weights
results = model.train(
    data="data_config.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    workers=8,
    amp=True,
    patience=100,
    project="runs/detect",
    name="train"
)
```

#### Training Results (50 Epochs):

| Metric | Final Value |
|--------|-------------|
| mAP50 (all) | 0.995 |
| mAP50-95 (all) | 0.846 |
| Box Loss | 0.580 |
| Class Loss | 0.331 |
| L1 Loss | 0.009 |

**Per-Class Performance:**
| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| check (0) | 0.991 | 1.000 | 0.995 | 0.955 |
| area (1) | 0.997 | 1.000 | 0.995 | 0.736 |

#### Inference Speed:
- Preprocess: 0.6ms
- Inference: 1.6ms
- Postprocess: 1.6ms
- **Total: ~3.8ms per image** (~260 FPS)

#### Output Files:
- `runs/detect/train/weights/best.pt` - Best model (mAP50-95)
- `runs/detect/train/weights/last.pt` - Last epoch model
- `runs/detect/train/results.png` - Training curves
- `runs/detect/train/labels.jpg` - Ground truth visualization
- `runs/detect/train/confusion_matrix.png` - Confusion matrix
- `runs/detect/train/*_curve.png` - PR, F1, P-conf, R-conf curves

## 🚀 Usage

### Training
```bash
# Via notebook
jupyter notebook yolo_detection.ipynb

# Or command line
yolo detect train data=data_config.yaml model=yolo26n.pt epochs=50 imgsz=640 batch=16 device=0
```

### Inference
```python
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Run inference on image
results = model.predict("path/to/image.jpg", imgsz=640, conf=0.25)

# Access results
for r in results:
    print(r.boxes.xywhn)  # Normalized xywh
    print(r.boxes.cls)    # Class IDs
    print(r.boxes.conf)   # Confidence scores
    r.show()              # Display with boxes
```

### Validation
```python
metrics = model.val(data="data_config.yaml", split="val")
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
```

## 📈 Key Achievements

- ✅ **High Accuracy**: mAP50 = 0.995, mAP50-95 = 0.846
- ✅ **Fast Inference**: ~3.8ms/image on RTX 3060
- ✅ **Lightweight Model**: Only 2.5M parameters (5.4MB)
- ✅ **Robust Detection**: Perfect recall (1.0) on both classes
- ✅ **Production Ready**: Exported weights ready for deployment

## 🔄 Pipeline Summary

```
LabelMe JSON Annotations
        │
        ▼
labelme_to_yolo_format.ipynb
        │
        ├───▶ YOLO format labels (.txt)
        │
        └───▶ Resized 640×640 Grayscale Images
                    │
                    ▼
            data_config.yaml
                    │
                    ▼
            yolo_detection.ipynb (Training)
                    │
                    ▼
            best.pt (Trained Model)
                    │
                    ▼
            Inference / Deployment
```

## 📝 Notes

1. **LabelMe Format Assumption**: The conversion script assumes exactly 2 shapes per image in fixed order (check first, area second). Modify if your annotation format differs.

2. **Grayscale Input**: Images are converted to single-channel grayscale. The model expects 1-channel input (adapted from 3-channel pre-trained weights).

3. **Transfer Learning**: Pre-trained COCO weights (yolo26n.pt) were fine-tuned. 606/708 layers transferred successfully.

4. **Class Imbalance**: Both classes have equal instances (89 each in val), balanced dataset.

5. **Early Stopping**: Patience=100 prevents overfitting; training stopped at epoch 50 (max epochs reached).

## 📄 License

Internal project for HamrahTel. All rights reserved.

---

*Documentation generated from source code and training logs.*