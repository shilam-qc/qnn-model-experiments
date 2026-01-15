# 📚 Model Documentation

Detailed information about each supported model in the Multi-Model NPU Inference project.

## Table of Contents
- [YOLOv8 Object Detection](#yolov8-object-detection)
- [YOLOv8-Pose Detection](#yolov8-pose-detection)
- [MobileNetV2 Classification](#mobilenetv2-classification)
- [ResNet50 Classification](#resnet50-classification)
- [YOLOv8-Seg Segmentation](#yolov8-seg-segmentation)

---

## YOLOv8 Object Detection

### Overview
YOLOv8 is the latest version of the YOLO (You Only Look Once) family, offering state-of-the-art object detection performance.

### Specifications
- **Input Size**: 640x640 RGB
- **Input Format**: Float32, normalized [0, 1]
- **Output**: Bounding boxes, class labels, confidence scores
- **Classes**: 80 COCO classes
- **NPU Performance**: 50-65 FPS
- **CPU Performance**: ~2 FPS

### Quantization
```powershell
python quantize_yolov8.py
```

### Inference
```powershell
python unified_inference.py --task detection
```

### Use Cases
- Real-time object detection
- Surveillance systems
- Robotics and autonomous vehicles
- Retail analytics

---

## YOLOv8-Pose Detection

### Overview
YOLOv8-Pose extends YOLOv8 to detect human poses with 17 COCO keypoints per person.

### Specifications
- **Input Size**: 640x640 RGB
- **Input Format**: Float32, normalized [0, 1]
- **Output**: Person bounding boxes + 17 keypoints (x, y, confidence)
- **Keypoints**: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
- **NPU Performance**: 40-50 FPS
- **CPU Performance**: ~1.5 FPS

### COCO Keypoints
```
0: nose
1: left_eye
2: right_eye
3: left_ear
4: right_ear
5: left_shoulder
6: right_shoulder
7: left_elbow
8: right_elbow
9: left_wrist
10: right_wrist
11: left_hip
12: right_hip
13: left_knee
14: right_knee
15: left_ankle
16: right_ankle
```

### Quantization
```powershell
python scripts/quantize_yolov8_pose.py
```

### Inference
```powershell
python unified_inference.py --task pose
```

### Use Cases
- Fitness and exercise tracking
- Gesture recognition
- Sports analysis
- Human-computer interaction
- Healthcare monitoring

---

## MobileNetV2 Classification

### Overview
MobileNetV2 is a lightweight convolutional neural network optimized for mobile and edge devices, trained on ImageNet.

### Specifications
- **Input Size**: 224x224 RGB
- **Input Format**: Float32, ImageNet normalized
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]
- **Output**: 1000 ImageNet class probabilities
- **NPU Performance**: 100+ FPS
- **CPU Performance**: ~5 FPS

### ImageNet Classes
The model can classify 1000 different object categories including:
- Animals (dogs, cats, birds, etc.)
- Vehicles (cars, trucks, airplanes, etc.)
- Objects (furniture, electronics, tools, etc.)
- Food items
- And many more

### Quantization
```powershell
python scripts/quantize_mobilenetv2.py
```

### Inference
```powershell
python unified_inference.py --task classification
```

### Use Cases
- Image classification
- Content moderation
- Product recognition
- Scene understanding
- Mobile applications

---

## ResNet50 Classification

### Overview
ResNet50 is a deep residual network with 50 layers, offering higher accuracy than MobileNetV2 at the cost of more computation.

### Specifications
- **Input Size**: 224x224 RGB
- **Input Format**: Float32, ImageNet normalized
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]
- **Output**: 1000 ImageNet class probabilities
- **NPU Performance**: 60-80 FPS
- **CPU Performance**: ~2 FPS

### Advantages over MobileNetV2
- Higher accuracy on complex images
- Better feature extraction
- More robust to variations

### Quantization
```powershell
python scripts/quantize_resnet50.py
```

### Inference
```powershell
python unified_inference.py --task classification --model models/resnet50_qdq.onnx
```

### Use Cases
- High-accuracy image classification
- Medical image analysis
- Quality control in manufacturing
- Scientific research
- Fine-grained classification tasks

---

## YOLOv8-Seg Segmentation

### Overview
YOLOv8-Seg combines object detection with instance segmentation, providing pixel-level masks for each detected object.

### Specifications
- **Input Size**: 640x640 RGB
- **Input Format**: Float32, normalized [0, 1]
- **Output**: Bounding boxes + segmentation masks
- **Classes**: 80 COCO classes
- **NPU Performance**: 30-40 FPS
- **CPU Performance**: ~1 FPS

### Output Format
- Detection boxes with class labels and confidence
- Pixel-level segmentation masks for each instance
- Mask resolution: Typically 160x160, upsampled to match input

### Quantization
```powershell
python scripts/quantize_yolov8_seg.py
```

### Inference
```powershell
python unified_inference.py --task segmentation
```

### Use Cases
- Medical image segmentation
- Autonomous driving (lane detection, object boundaries)
- Precise object isolation for editing
- Agricultural monitoring
- Industrial inspection

---

## Performance Comparison

| Model | Task | Input Size | NPU FPS | CPU FPS | Speedup | Accuracy |
|-------|------|-----------|---------|---------|---------|----------|
| YOLOv8 | Detection | 640x640 | 50-65 | ~2 | 25-30x | High |
| YOLOv8-Pose | Pose | 640x640 | 40-50 | ~1.5 | 25-30x | High |
| MobileNetV2 | Classification | 224x224 | 100+ | ~5 | 20x | Good |
| ResNet50 | Classification | 224x224 | 60-80 | ~2 | 30-40x | Very High |
| YOLOv8-Seg | Segmentation | 640x640 | 30-40 | ~1 | 30-40x | High |

---

## Model Selection Guide

### Choose YOLOv8 Detection if:
- You need real-time object detection
- You want to detect multiple objects simultaneously
- Bounding boxes are sufficient (no need for masks)

### Choose YOLOv8-Pose if:
- You need human pose estimation
- You're building fitness or sports applications
- You need skeletal tracking

### Choose MobileNetV2 if:
- You need fast classification
- You're building mobile/edge applications
- Speed is more important than accuracy

### Choose ResNet50 if:
- You need high-accuracy classification
- You have complex or fine-grained categories
- You can afford slightly lower FPS

### Choose YOLOv8-Seg if:
- You need pixel-level object boundaries
- You're doing medical imaging or precise editing
- You need both detection and segmentation

---

## Quantization Details

All models use **INT8 QDQ (Quantize-Dequantize)** format for NPU acceleration:

- **Activation Type**: QUInt8 (Unsigned 8-bit)
- **Weight Type**: QUInt8 (Unsigned 8-bit)
- **Calibration Method**: MinMax
- **Calibration Dataset**: COCO128 (128 images)
- **ONNX Opset**: 17

### Quantization Benefits
- 4x smaller model size
- 25-40x faster inference on NPU
- Minimal accuracy loss (<2% typically)
- Lower power consumption

---

## Custom Model Integration

To add your own model:

1. Create a quantization script in `scripts/`
2. Implement preprocessing/postprocessing
3. Create an inference script in `inference/`
4. Add to `unified_inference.py`
5. Update documentation

See existing scripts as templates.

---

## References

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [PyTorch Vision Models](https://pytorch.org/vision/stable/models.html)
- [ONNX Runtime QNN](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- [COCO Dataset](https://cocodataset.org/)
- [ImageNet](https://www.image-net.org/)
