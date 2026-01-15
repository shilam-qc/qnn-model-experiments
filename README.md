# ⚡ Multi-Model AI on Snapdragon X Elite (NPU)

Run **multiple AI models** at high FPS on the Windows ARM Hexagon NPU using ONNX Runtime QNN Execution Provider.

![NPU Ready](https://img.shields.io/badge/NPU-Ready-green) ![Platform](https://img.shields.io/badge/Platform-Snapdragon_X_Elite-blue) ![Models](https://img.shields.io/badge/Models-4_Types-orange)

## 🎯 Supported Models

| Model | Task | Expected FPS (NPU) | Status |
|-------|------|-------------------|--------|
| **YOLOv8** | Object Detection | 50-65 FPS | ✅ Ready |
| **YOLOv8-Pose** | Pose Detection | 40-50 FPS | ✅ Ready |
| **MobileNetV2** | Classification | 100+ FPS | ✅ Ready |
| **ResNet50** | Classification | 60-80 FPS | ✅ Ready |
| **YOLOv8-Seg** | Segmentation | 30-40 FPS | ✅ Ready |

## 📋 Prerequisites

- **Device**: Surface Laptop 7 / Surface Pro 11 / Any generic **Snapdragon X Elite** device
- **OS**: Windows 11 on Arm
- **Drivers**: Latest Qualcomm NPU drivers (Windows Update handles this)
- **Python**: Python 3.10+ (ARM64 Native)

## 🚀 Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Download Calibration Data (First Time Only)
```powershell
python download_assets.py
```

### 3. Run Inference

#### Option A: Unified Interface (Recommended)
```powershell
# Object Detection (YOLOv8)
python unified_inference.py --task detection --duration 30

# Pose Detection (YOLOv8-Pose)
python unified_inference.py --task pose --duration 30

# Classification (MobileNetV2)
python unified_inference.py --task classification --duration 30

# Classification (ResNet50)
python unified_inference.py --task classification --model models/resnet50_qdq.onnx

# Segmentation (YOLOv8-Seg)
python unified_inference.py --task segmentation --duration 30
```

#### Option B: Individual Scripts
```powershell
# Object Detection
python npu_realtime_inference.py --duration 30

# Pose Detection
python inference/npu_pose_inference.py --duration 30

# Classification
python inference/npu_classification_inference.py --duration 30

# Segmentation
python inference/npu_segmentation_inference.py --duration 30
```

## 🛠️ Model Preparation

### Pre-quantized Models
The repository includes a pre-quantized YOLOv8 detection model. For other models, you need to quantize them first:

```powershell
# YOLOv8-Pose
python scripts/quantize_yolov8_pose.py

# MobileNetV2
python scripts/quantize_mobilenetv2.py

# ResNet50
python scripts/quantize_resnet50.py

# YOLOv8-Seg
python scripts/quantize_yolov8_seg.py
```

### Custom Models
You can quantize custom models by modifying the quantization scripts in the `scripts/` directory.

## 📂 Project Structure

```
Object-Detection-QNN/
├── models/                          # Quantized models directory
│   ├── yolov8s-pose_qdq.onnx       # Pose detection model
│   ├── mobilenetv2_qdq.onnx        # Classification model
│   ├── resnet50_qdq.onnx           # Classification model
│   └── yolov8s-seg_qdq.onnx        # Segmentation model
├── scripts/                         # Quantization scripts
│   ├── quantize_yolov8_pose.py
│   ├── quantize_mobilenetv2.py
│   ├── quantize_resnet50.py
│   └── quantize_yolov8_seg.py
├── inference/                       # Inference scripts
│   ├── npu_pose_inference.py
│   ├── npu_classification_inference.py
│   └── npu_segmentation_inference.py
├── unified_inference.py             # Unified multi-model interface
├── npu_realtime_inference.py        # Original YOLOv8 detection
├── technical_benchmark.py           # Benchmarking tool
├── quantize_yolov8.py              # Original YOLOv8 quantization
├── download_assets.py              # Dataset downloader
├── yolov8s_qdq.onnx                # Pre-quantized YOLOv8 model
└── requirements.txt                # Python dependencies
```

## 🔬 How It Works

The pipeline for each model:

1. **Export**: PyTorch/Pretrained model → ONNX (Float32)
2. **Quantize**: ONNX → **INT8 QDQ ONNX** (using calibration data)
3. **Inference**: Uses `onnxruntime-qnn` with `QnnHtp.dll` backend for NPU acceleration

### Performance Comparison

| Model | CPU FPS | NPU FPS | Speedup |
|-------|---------|---------|---------|
| YOLOv8 Detection | ~2 | 50-65 | 25-30x |
| YOLOv8-Pose | ~1.5 | 40-50 | 25-30x |
| MobileNetV2 | ~5 | 100+ | 20x |
| ResNet50 | ~2 | 60-80 | 30-40x |
| YOLOv8-Seg | ~1 | 30-40 | 30-40x |

## 📊 Monitoring NPU Usage

Open **Task Manager** (`Ctrl+Shift+Esc`) → **Performance** → **NPU** to see real-time NPU utilization during inference.

## 🎨 Model-Specific Details

### Object Detection (YOLOv8)
- **Input**: 640x640 RGB images
- **Output**: Bounding boxes + class labels + confidence scores
- **Use Cases**: General object detection, surveillance, robotics

### Pose Detection (YOLOv8-Pose)
- **Input**: 640x640 RGB images
- **Output**: 17 COCO keypoints per person (nose, eyes, shoulders, etc.)
- **Use Cases**: Fitness tracking, gesture recognition, sports analysis

### Classification (MobileNetV2 / ResNet50)
- **Input**: 224x224 RGB images (ImageNet normalized)
- **Output**: 1000 ImageNet class probabilities
- **Use Cases**: Image classification, scene understanding, content moderation

### Segmentation (YOLOv8-Seg)
- **Input**: 640x640 RGB images
- **Output**: Bounding boxes + pixel-level segmentation masks
- **Use Cases**: Medical imaging, autonomous driving, precise object isolation

## 🐛 Troubleshooting

### "QNN execution provider not found"
```powershell
pip uninstall onnxruntime onnxruntime-gpu
pip install --force-reinstall onnxruntime-qnn
```

### Low FPS (~2-3 FPS)
- NPU failed to initialize, falling back to CPU
- Check console output for QNN warnings
- Ensure `QnnHtp.dll` is accessible

### Model Not Found
Run the appropriate quantization script first:
```powershell
python scripts/quantize_<model_name>.py
```

### Out of Memory
- Reduce batch size (already set to 1)
- Use smaller model variants (e.g., `yolov8n` instead of `yolov8s`)

## 🔧 Advanced Usage

### Custom Duration
```powershell
python unified_inference.py --task detection --duration 60
```

### Custom Data Directory
```powershell
python unified_inference.py --task pose --data path/to/images
```

### Show Predictions (Classification)
```powershell
python unified_inference.py --task classification --show-predictions
```

### Benchmark Mode
```powershell
python technical_benchmark.py --model models/mobilenetv2_qdq.onnx --backend qnn
```

## 📚 Additional Resources

- [ONNX Runtime QNN Documentation](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Qualcomm AI Hub](https://aihub.qualcomm.com/)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add support for new models
- Improve quantization techniques
- Optimize inference performance
- Fix bugs and improve documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Qualcomm for the Snapdragon X Elite NPU
- Microsoft for ONNX Runtime QNN support
- Ultralytics for YOLOv8 models
- PyTorch team for torchvision models

---

**Note**: This is a research/development project. Performance may vary based on hardware, drivers, and model configurations.

https://github.com/user-attachments/assets/b14b2a68-00e0-45aa-a753-e13b5d37b89f
