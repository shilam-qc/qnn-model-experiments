## How the Models Work

### Original YOLOv8 (Already Present)

- `yolov8s.onnx` - Pre-exported float32 model
- `yolov8s_qdq.onnx` - Pre-quantized INT8 model (ready to use)

### New Models (Need to be Generated)

The 4 new models are __not included__ in the repository because:

1. They're large files (50-200 MB each)
2. They need to be downloaded from their respective sources first
3. They require quantization with your local calibration data

## Generation Process

Each model follows this workflow:

### 1. __YOLOv8-Pose__

```powershell
python scripts/quantize_yolov8_pose.py
```

- Downloads `yolov8s-pose.pt` from Ultralytics (if not present)
- Exports to ONNX format
- Quantizes to INT8 QDQ
- Saves to `models/yolov8s-pose_qdq.onnx`

### 2. __MobileNetV2__

```powershell
python scripts/quantize_mobilenetv2.py
```

- Downloads pretrained weights from PyTorch/torchvision
- Exports to ONNX format
- Quantizes to INT8 QDQ
- Saves to `models/mobilenetv2_qdq.onnx`

### 3. __ResNet50__

```powershell
python scripts/quantize_resnet50.py
```

- Downloads pretrained weights from PyTorch/torchvision
- Exports to ONNX format
- Quantizes to INT8 QDQ
- Saves to `models/resnet50_qdq.onnx`

### 4. __YOLOv8-Seg__

```powershell
python scripts/quantize_yolov8_seg.py
```

- Downloads `yolov8s-seg.pt` from Ultralytics (if not present)
- Exports to ONNX format
- Quantizes to INT8 QDQ
- Saves to `models/yolov8s-seg_qdq.onnx`

## Automatic Handling

The good news is that __`run_multi_model.bat` handles this automatically__! When you select a model to run:

1. It checks if the quantized model exists in `models/` directory
2. If missing, it automatically runs the quantization script
3. Then proceeds with inference

So you can simply run:

```powershell
run_multi_model.bat
```

And select option 2, 3, 4, or 5 - it will download and quantize the model automatically on first use.

## Manual Generation (All at Once)

If you want to generate all models at once:

```powershell
# Run the batch file and select "Q" for Quantize, then "5" for All Models
run_multi_model.bat
```

Or manually:

```powershell
python scripts/quantize_yolov8_pose.py
python scripts/quantize_mobilenetv2.py
python scripts/quantize_resnet50.py
python scripts/quantize_yolov8_seg.py
```

## Storage Requirements

After generation, expect these approximate sizes:

- YOLOv8-Pose: ~25 MB (quantized)
- MobileNetV2: ~9 MB (quantized)
- ResNet50: ~25 MB (quantized)
- YOLOv8-Seg: ~27 MB (quantized)

__Total: ~86 MB__ for all 4 new models
