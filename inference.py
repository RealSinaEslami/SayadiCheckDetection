#!/usr/bin/env python3
"""
Inference script for Handwritten Text Detection (Check & Area)
Usage: python inference.py --image path/to/image.jpg --model runs/detect/train/weights/best.pt
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class HandwrittenDetector:
    """Detector for check and area in handwritten documents."""
    
    CLASS_NAMES = {0: "check", 1: "area"}
    CLASS_COLORS = {0: (0, 255, 0), 1: (255, 0, 0)}  # Green for check, Blue for area
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.7):
        """
        Initialize detector.
        
        Args:
            model_path: Path to .pt model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        print(f"Loaded model: {model_path}")
        print(f"Classes: {self.CLASS_NAMES}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image: grayscale + resize to 640x640."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        resized = cv2.resize(gray, (640, 640))
        # Convert back to 3-channel for YOLO (expects 3 channels)
        return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    
    def predict(self, image: np.ndarray) -> list:
        """
        Run inference on image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            List of detections: [{'class', 'class_name', 'confidence', 'bbox': [x, y, w, h]}, ...]
            bbox in normalized coordinates (0-1)
        """
        processed = self.preprocess(image)
        results = self.model.predict(
            processed,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        detections = []
        for r in results:
            if r.boxes is not None:
                boxes = r.boxes.xywhn.cpu().numpy()  # normalized xywh
                classes = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                
                for box, cls, conf in zip(boxes, classes, confs):
                    detections.append({
                        'class': int(cls),
                        'class_name': self.CLASS_NAMES.get(cls, f"class_{cls}"),
                        'confidence': float(conf),
                        'bbox': box.tolist()  # [x_center, y_center, width, height] normalized
                    })
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        """Draw bounding boxes on image."""
        vis = image.copy()
        h, w = vis.shape[:2]
        
        for det in detections:
            x_center, y_center, width, height = det['bbox']
            
            # Convert normalized to pixel coordinates
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            
            color = self.CLASS_COLORS.get(det['class'], (255, 255, 255))
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            
            # Draw box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(vis, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis
    
    def detect_image(self, image_path: str, output_path: str = None, show: bool = False):
        """Run detection on single image file."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        detections = self.predict(image)
        vis = self.draw_detections(image, detections)
        
        if output_path:
            cv2.imwrite(output_path, vis)
            print(f"Saved result to: {output_path}")
        
        if show:
            cv2.imshow("Detection", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return detections, vis
    
    def detect_directory(self, input_dir: str, output_dir: str):
        """Run detection on all images in directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
        
        print(f"Processing {len(image_files)} images...")
        
        all_results = {}
        for img_file in image_files:
            try:
                detections, vis = self.detect_image(str(img_file))
                out_file = output_path / f"det_{img_file.name}"
                cv2.imwrite(str(out_file), vis)
                all_results[img_file.name] = detections
                print(f"  {img_file.name}: {len(detections)} detections")
            except Exception as e:
                print(f"  Error processing {img_file.name}: {e}")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Handwritten Check/Area Detection")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--dir", type=str, help="Path to input directory")
    parser.add_argument("--model", type=str, default="runs/detect/train/weights/best.pt",
                        help="Path to model weights")
    parser.add_argument("--output", type=str, help="Output path (image or directory)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold")
    parser.add_argument("--show", action="store_true", help="Show result window")
    
    args = parser.parse_args()
    
    if not args.image and not args.dir:
        parser.error("Either --image or --dir must be provided")
    
    detector = HandwrittenDetector(args.model, args.conf, args.iou)
    
    if args.image:
        detections, _ = detector.detect_image(args.image, args.output, args.show)
        print(f"\nDetections ({len(detections)}):")
        for det in detections:
            print(f"  {det['class_name']}: conf={det['confidence']:.3f}, "
                  f"bbox={['{:.3f}'.format(x) for x in det['bbox']]}")
    
    if args.dir:
        results = detector.detect_directory(args.dir, args.output or "output")
        print(f"\nProcessed {len(results)} images")


if __name__ == "__main__":
    main()