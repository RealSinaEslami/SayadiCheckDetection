# Technical Documentation

## Overview

This document provides detailed technical documentation for the Handwritten Text Detection project, covering all code modules, functions, data flows, and architectural decisions.

---

## Module 1: LabelMe to YOLO Converter (`labelme_to_yolo_format.ipynb`)

### Purpose
Converts LabelMe annotation format (JSON) to YOLO detection format (TXT) and preprocesses images.

### Dependencies
```python
import os           # File system operations
import shutil       # File copying/moving
import json         # JSON parsing
import cv2 as cv    # Image processing
```

### Configuration Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `images_path` | Source images directory | `yolo_dataset/images/` |
| `labels_path` | LabelMe JSON annotations directory | `yolo_dataset/labels/` |
| `save_txt_path` | Output YOLO labels directory | `yolo_dataset/text_labels/` |
| `old_images` | Source folder for raw images | `old_images` |

### Core Functions

#### 1. `parse_labelme_json(json_path, img_shape)`
**Input**: Path to LabelMe JSON file, image shape (height, width)
**Output**: List of YOLO format annotations `[[class_id, x_center, y_center, width, height], ...]`

**Algorithm**:
```python
def parse_labelme_json(json_path, img_shape):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    h, w = img_shape  # height, width
    annotations = []
    
    for shape in data['shapes']:
        # LabelMe stores rectangle as 2 corner points
        points = shape['points']
        x0, y0 = points[0]
        x1, y1 = points[1]
        
        # Class mapping (hardcoded based on label name)
        class_id = 0 if shape['label'] == 'check' else 1
        
        # Convert to YOLO format (normalized)
        x_center = ((x0 + x1) / 2) / w
        y_center = ((y0 + y1) / 2) / h
        width = abs(x1 - x0) / w
        height = abs(y1 - y0) / h
        
        annotations.append([class_id, x_center, y_center, width, height])
    
    return annotations
```

#### 2. `process_dataset()`
**Main processing loop** - iterates through all label files:
```python
def process_dataset():
    labels_list = os.listdir(labels_path)
    
    for i, label_file in enumerate(labels_list):
        try:
            # Load corresponding image to get dimensions
            img_file = images_list[i]
            img = cv.imread(os.path.join(images_path, img_file))
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Parse annotations
            annotations = parse_labelme_json(
                os.path.join(labels_path, label_file), 
                (h, w)
            )
            
            # Save YOLO format
            save_path = os.path.join(save_txt_path, label_file.replace('.json', '.txt'))
            with open(save_path, 'w') as f:
                for ann in annotations:
                    f.write(' '.join(map(str, ann)) + '\n')
            
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
```

#### 3. `resize_images()`
**Preprocesses images** to 640×640 grayscale:
```python
def resize_images():
    src = os.path.join("yolo_dataset", old_images)
    dst = os.path.join("yolo_dataset", "images")
    
    for img_file in os.listdir(src):
        img = cv.imread(os.path.join(src, img_file))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        resized = cv.resize(gray, (640, 640), interpolation=cv.INTER_AREA)
        cv.imwrite(os.path.join(dst, img_file), resized)
```

### Data Flow
```
LabelMe JSON (polygon) → Parse 2 corner points → Calculate bbox → Normalize → YOLO TXT
Raw Image (any size)   → Grayscale → Resize 640×640 → Save
```

### Assumptions & Limitations
1. **Fixed class order**: Assumes `shapes[0]` = check, `shapes[1]` = area
2. **Rectangle annotations**: Expects exactly 2 points per shape (rectangle corners)
3. **Matching filenames**: Image and label files must have same base name
4. **No validation**: No check for annotation validity or image-label correspondence

### Recommended Improvements
```python
# 1. Dynamic class mapping
CLASS_MAP = {'check': 0, 'area': 1}

# 2. Support polygons (not just rectangles)
def polygon_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)

# 3. Validate image-label pairing
assert os.path.exists(img_path), f"Missing image for {label_file}"

# 4. Handle multiple shapes per class
# 5. Add logging instead of print
```

---

## Module 2: YOLO Training (`yolo_detection.ipynb`)

### Purpose
Train, validate, and evaluate YOLOv8n model for handwritten check/area detection.

### Model Architecture: YOLO26n (Ultralytics YOLOv8n)

#### Backbone (Feature Extraction)
| Layer | Module | In/Out Channels | Params | Description |
|-------|--------|-----------------|--------|-------------|
| 0 | Conv | 3→16, s=2 | 464 | Stem conv 3×3 |
| 1 | Conv | 16→32, s=2 | 4,672 | Downsample |
| 2 | C3k2 | 32→64 | 6,640 | CSP block k=2 |
| 3 | Conv | 64→64, s=2 | 36,992 | Downsample |
| 4 | C3k2 | 64→128 | 26,080 | CSP block |
| 5 | Conv | 128→128, s=2 | 147,712 | Downsample |
| 6 | C3k2 | 128→128 | 87,040 | CSP block (shortcut) |
| 7 | Conv | 128→256, s=2 | 295,424 | Downsample |
| 8 | C3k2 | 256→256 | 346,112 | CSP block (shortcut) |
| 9 | SPPF | 256→256 | 164,608 | Spatial Pyramid Pooling |
| 10 | C2PSA | 256→256 | 249,728 | Attention block |

#### Neck (Feature Fusion)
| Layer | Module | Inputs | Params | Description |
|-------|--------|--------|--------|-------------|
| 11 | Upsample | - | 0 | ×2 nearest neighbor |
| 12 | Concat | [11, 6] | 0 | Skip connection |
| 13 | C3k2 | 384→128 | 119,808 | Fusion |
| 14 | Upsample | - | 0 | ×2 |
| 15 | Concat | [14, 4] | 0 | Skip connection |
| 16 | C3k2 | 256→64 | 34,304 | Fusion |
| 17 | Conv | 64→64, s=2 | 36,992 | Downsample |
| 18 | Concat | [17, 13] | 0 | Skip connection |
| 19 | C3k2 | 192→128 | 95,232 | Fusion |
| 20 | Conv | 128→128, s=2 | 147,712 | Downsample |
| 21 | Concat | [20, 10] | 0 | Skip connection |
| 22 | C3k2 | 384→256 | 463,104 | Fusion (attention) |

#### Head (Detection)
| Layer | Module | Inputs | Params | Description |
|-------|--------|--------|--------|-------------|
| 23 | Detect | [16, 19, 22] | 241,956 | Multi-scale detection |

**Total**: 2,504,580 params, 5.9 GFLOPs

### Training Configuration

#### Hyperparameters (`train_args`)
```python
train_args = {
    # Data
    'data': 'data_config.yaml',
    'imgsz': 640,
    'batch': 16,
    'fraction': 1.0,
    
    # Model
    'model': 'yolo26n.pt',
    'task': 'detect',
    'pretrained': True,
    
    # Training
    'epochs': 50,
    'patience': 100,
    'device': 0,  # CUDA:0
    'workers': 8,
    
    # Optimization
    'optimizer': 'auto',  # AdamW selected
    'lr0': 0.01,          # Ignored when auto
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'cos_lr': False,
    
    # Loss Weights
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    
    # Augmentation
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.0,
    'copy_paste': 0.0,
    'erasing': 0.4,
    'auto_augment': 'randaugment',
    
    # Regularization
    'dropout': 0.0,
    'label_smoothing': 0.0,
    'nms': False,
    
    # Logging
    'project': 'runs/detect',
    'name': 'train',
    'exist_ok': False,
    'plots': True,
    'save': True,
    'save_period': -1,
    'verbose': True,
}
```

### Loss Function
YOLOv8 uses **Composite Loss**:
```
L_total = λ_box * L_box + λ_cls * L_cls + λ_dfl * L_dfl

Where:
- L_box: CIoU Loss (Complete IoU) for bounding box regression
- L_cls: Binary Cross Entropy for classification
- L_dfl: Distribution Focal Loss for box distribution
```

### Training Loop (Internal Ultralytics)
```python
# Pseudocode of training epoch
for epoch in range(epochs):
    model.train()
    
    for batch in train_loader:
        # Forward
        preds = model(batch['img'])
        
        # Compute loss
        loss_box, loss_cls, loss_dfl = criterion(preds, batch['targets'])
        loss = box_weight * loss_box + cls_weight * loss_cls + dfl_weight * loss_dfl
        
        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Validation
    if epoch % val_interval == 0:
        metrics = validate(model, val_loader)
        
    # Logging & Checkpointing
    log_metrics(metrics)
    save_checkpoint(model, optimizer, epoch, metrics)
```

### Validation Metrics

#### Computation (per class)
```python
# IoU thresholds: 0.5:0.95 (10 points)
# mAP = mean(AP@0.5, AP@0.55, ..., AP@0.95)

# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)
# F1 = 2 * P * R / (P + R)

# AP = area under P-R curve (101-point interpolation)
# mAP = mean AP across classes
```

#### Final Results (Epoch 50)
```
Class 'check' (0):
  Precision: 0.991
  Recall:    1.000
  mAP50:     0.995
  mAP50-95:  0.955

Class 'area' (1):
  Precision: 0.997
  Recall:    1.000
  mAP50:     0.995
  mAP50-95:  0.736

Overall:
  mAP50:     0.995
  mAP50-95:  0.846
```

### Model Export & Inference

#### Export Formats Supported
```python
model.export(format='onnx')      # ONNX
model.export(format='torchscript') # TorchScript
model.export(format='engine')     # TensorRT
model.export(format='coreml')     # CoreML
model.export(format='tflite')     # TensorFlow Lite
```

#### Inference Pipeline
```python
def predict(image_path, model_path='best.pt', conf=0.25, iou=0.7):
    model = YOLO(model_path)
    
    # Preprocess
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (640, 640))
    
    # Inference
    results = model(img, conf=conf, iou=iou)
    
    # Postprocess
    detections = []
    for r in results:
        boxes = r.boxes.xywhn.cpu().numpy()  # normalized xywh
        classes = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        
        for box, cls, conf in zip(boxes, classes, confs):
            detections.append({
                'class': int(cls),
                'class_name': 'check' if cls == 0 else 'area',
                'confidence': float(conf),
                'bbox': box.tolist()  # [x_center, y_center, width, height]
            })
    
    return detections
```

---

## Module 3: Dataset Configuration (`data_config.yaml`)

### Structure
```yaml
path: C:/Users/Sina/Desktop/HamrahTel/Handwritten/yolo_dataset/  # Root dataset path
train: images/train      # Relative to path: training images
val: images/val          # Relative to path: validation images
# test: images/test      # Optional: test images

names:                   # Class name mapping
  0: check               # Class 0 → "check"
  1: area                # Class 1 → "area"
```

### Path Resolution
```
Absolute train path: {path}/{train} = C:/.../yolo_dataset/images/train/
Absolute val path:   {path}/{val}   = C:/.../yolo_dataset/images/val/
```

### Requirements
- Images: `.jpg`, `.png`, `.jpeg`, `.bmp`, `.tiff`
- Labels: `.txt` files with same stem as images in `labels/{train,val}/`
- Label format: `class_id x_center y_center width height` (normalized 0-1)

---

## Data Pipeline Details

### 1. Data Loading (Ultralytics DataLoader)
```python
# Train transforms (applied per image)
transforms = Compose([
    Mosaic(p=1.0),           # 4-image mosaic
    RandomPerspective(),     # Perspective transform
    Mixup(p=0.0),            # Disabled
    CopyPaste(p=0.0),        # Disabled
    Albumentations([
        HSV(h=0.015, s=0.7, v=0.4),
        RandomFlip(p=0.5),
        RandomScale(scale=0.5),
        RandomTranslate(translate=0.1),
    ]),
    Letterbox(new_shape=640), # Resize with padding
    Normalize(),              # /255
    ToTensor(),
])
```

### 2. Collate Function
```python
def collate_fn(batch):
    # Batch dict with keys: 'img', 'cls', 'bboxes', 'batch_idx'
    imgs = torch.stack([b['img'] for b in batch])
    cls = torch.cat([b['cls'] for b in batch])
    bboxes = torch.cat([b['bboxes'] for b in batch])
    batch_idx = torch.cat([torch.full((len(b['cls']),), i) for i, b in enumerate(batch)])
    return {'img': imgs, 'cls': cls, 'bboxes': bboxes, 'batch_idx': batch_idx}
```

### 3. Target Format for Loss
```python
# Targets: [batch_idx, class_id, x_center, y_center, width, height]
# All normalized to [0, 1]
targets = torch.tensor([
    [0, 0, 0.5, 0.5, 0.2, 0.3],  # image 0, class 0
    [0, 1, 0.3, 0.7, 0.1, 0.2],  # image 0, class 1
    [1, 0, 0.6, 0.4, 0.15, 0.25], # image 1, class 0
])
```

---

## Performance Analysis

### Computational Requirements
| Component | Specification |
|-----------|---------------|
| GPU Memory (train) | ~3 GB |
| GPU Memory (inference) | ~1 GB |
| Training Time (50 epochs) | ~3.7 minutes |
| Inference Latency | 1.6 ms/image |
| Throughput | ~625 img/sec (batch=1) |
| Model Size | 5.4 MB |

### Scaling Recommendations
```python
# For higher accuracy (trade speed)
model = YOLO('yolo11s.pt')   # Small: 9.5M params, 28 GFLOPs
model = YOLO('yolo11m.pt')   # Medium: 25.8M params, 78 GFLOPs

# For faster inference
model.export(format='engine', half=True)  # TensorRT FP16
# Expected: ~0.5 ms/image on RTX 3060
```

---

## Error Handling & Debugging

### Common Issues

#### 1. "No labels found" Error
```python
# Check label files exist and have content
for label_file in os.listdir('yolo_dataset/labels/train'):
    path = os.path.join('yolo_dataset/labels/train', label_file)
    with open(path) as f:
        lines = f.readlines()
    if not lines:
        print(f"Empty label: {label_file}")
```

#### 2. Class Index Mismatch
```python
# Verify data_config.yaml matches actual labels
# Labels must use 0 and 1 only
unique_classes = set()
for label_file in os.listdir('yolo_dataset/labels/train'):
    with open(os.path.join('yolo_dataset/labels/train', label_file)) as f:
        for line in f:
            unique_classes.add(int(line.split()[0]))
print(f"Classes found: {unique_classes}")  # Should be {0, 1}
```

#### 3. Image-Label Mismatch
```python
# Verify pairing
images = set(f.split('.')[0] for f in os.listdir('yolo_dataset/images/train'))
labels = set(f.split('.')[0] for f in os.listdir('yolo_dataset/labels/train'))
missing_labels = images - labels
missing_images = labels - images
```

#### 4. CUDA OOM
```python
# Reduce batch size
batch = 8  # or 4

# Enable gradient accumulation
accumulate = 2  # Simulates batch=16 with batch=8

# Or use CPU (slow)
device = 'cpu'
```

---

## Extending the Project

### Adding New Classes
1. Update `data_config.yaml`:
```yaml
names:
  0: check
  1: area
  2: signature  # New class
```

2. Retrain from scratch or fine-tune:
```python
model = YOLO('best.pt')  # Continue training
model.train(data='data_config.yaml', epochs=20)
```

### Custom Augmentation
```python
from ultralytics.data.augment import Albumentations
import albumentations as A

custom_aug = A.Compose([
    A.GaussNoise(p=0.3),
    A.MotionBlur(p=0.2),
    A.RandomBrightnessContrast(p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Pass to trainer
trainer = YOLO('yolo26n.pt')
trainer.train(..., augment=custom_aug)
```

### Custom Loss Weights
```python
# For class imbalance
model.train(..., cls=1.0)  # Increase class loss weight

# For better localization
model.train(..., box=10.0)  # Increase box loss weight
```

---

## File Reference

| File | Description | Lines | Type |
|------|-------------|-------|------|
| `labelme_to_yolo_format.ipynb` | Annotation converter | 165 | Notebook |
| `yolo_detection.ipynb` | Training pipeline | 400+ | Notebook |
| `data_config.yaml` | Dataset config | 8 | YAML |
| `yolo26n.pt` | Pre-trained weights | - | Binary |
| `weights/yolo26n.pt` | Backup weights | - | Binary |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024 | Initial implementation |
| 1.1 | 2024 | Added technical documentation |

---

*This technical documentation covers all code modules in the Handwritten Text Detection project.*