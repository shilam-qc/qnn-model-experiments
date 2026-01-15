import argparse
from pathlib import Path
import numpy as np
import cv2
import onnx
import torch
import torchvision.models as models
from onnxruntime.quantization import (
    quantize_static,
    QuantFormat,
    QuantType,
    CalibrationMethod,
    CalibrationDataReader
)

class ImageNetDataReader(CalibrationDataReader):
    def __init__(self, image_dir, input_name, input_shape, size=100):
        self.image_dir = Path(image_dir)
        self.input_name = input_name
        self.input_shape = input_shape
        self.image_paths = list(self.image_dir.glob('*.jpg'))[:size]
        self.enum_data = None
        
        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        print(f"Found {len(self.image_paths)} images for calibration")

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.image_paths)
        
        try:
            image_path = next(self.enum_data)
            return {self.input_name: self.preprocess(image_path)}
        except StopIteration:
            return None

    def preprocess(self, image_path):
        # ResNet50 expects RGB, ImageNet normalized, (1, 3, 224, 224)
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w = self.input_shape[2], self.input_shape[3]
        img = cv2.resize(img, (w, h))
        
        # Normalize to 0-1
        img = img.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        img = (img - self.mean) / self.std
        
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dim
        return img

def export_to_onnx(output_path='resnet50.onnx'):
    print("Loading ResNet50 from torchvision...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    
    print("Exporting to ONNX...")
    dummy_input = torch.randn(1, 3, 224, 224)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {output_path}")
    return output_path

def quantize_model(onnx_path, calibration_data_dir, output_dir='models'):
    model = onnx.load(onnx_path)
    input_name = model.graph.input[0].name
    input_shape = (1, 3, 224, 224)

    output_path = Path(output_dir) / Path(onnx_path).name.replace('.onnx', '_qdq.onnx')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    dr = ImageNetDataReader(calibration_data_dir, input_name, input_shape, size=128)
    
    print(f"Quantizing {onnx_path} to QDQ format...")
    
    quantize_static(
        model_input=onnx_path,
        model_output=str(output_path),
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            'ActivationSymmetric': False,
            'WeightSymmetric': True 
        }
    )
    print(f"Quantized model saved to {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export and quantize ResNet50 for NPU')
    parser.add_argument('--model', type=str, default='resnet50.onnx', help='ONNX model path')
    parser.add_argument('--data_dir', type=str, default='datasets/coco128/images/train2017', 
                        help='Path to calibration images')
    parser.add_argument('--output_dir', type=str, default='models', help='Output directory for quantized model')
    args = parser.parse_args()

    # 1. Export to ONNX if needed
    if not Path(args.model).exists():
        print("Model not found. Exporting from torchvision...")
        onnx_path = export_to_onnx(args.model)
    else:
        print(f"Found existing {args.model}")
        onnx_path = args.model

    # 2. Quantize
    quantize_model(onnx_path, args.data_dir, args.output_dir)
    print("\n✅ ResNet50 quantization complete!")
    print(f"   Run inference with: python inference/npu_classification_inference.py --model models/resnet50_qdq.onnx")
