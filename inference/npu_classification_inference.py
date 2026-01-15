import time
import argparse
import numpy as np
import onnxruntime as ort
import cv2
from pathlib import Path
import sys
import json

def print_banner():
    print(r"""
   _____ __              _ _____           __  _          
  / ___// /__ ____ ___ _(_) _(_)_______ _/ /_(_)__  ___  
 / /__/ / _ `(_-<(_-</ / _/ / __/ __/ _ `/ __/ / _ \/ _ \ 
 \___/_/\_,_/___/___/_/_//_/\__/\__/\_,_/\__/_/\___/_//_/ 
     ON SNAPDRAGON X ELITE | HEXAGON NPU
    """)

# ImageNet class labels (top 5 will be shown)
def load_imagenet_labels():
    """Load ImageNet class labels"""
    # Simplified - in production, load from file
    return {i: f"class_{i}" for i in range(1000)}

def preprocess_imagenet(image_path, input_shape):
    """Preprocess for ImageNet models (MobileNetV2, ResNet50)"""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = input_shape[2], input_shape[3]
    img = cv2.resize(img, (w, h))
    
    # Normalize to 0-1
    img = img.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dim
    return img

def postprocess_classification(output, top_k=5):
    """Get top-k predictions"""
    if isinstance(output, list):
        output = output[0]
    
    # Flatten if needed
    if len(output.shape) > 2:
        output = output.reshape(output.shape[0], -1)
    
    # Get top-k indices
    top_indices = np.argsort(output[0])[-top_k:][::-1]
    top_probs = output[0][top_indices]
    
    # Apply softmax for probabilities
    exp_probs = np.exp(top_probs - np.max(top_probs))
    softmax_probs = exp_probs / np.sum(exp_probs)
    
    return list(zip(top_indices, softmax_probs))

def run_npu_inference(model_path, data_dir, duration=20, show_predictions=False):
    print_banner()
    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] Checking Execution Providers...")
    
    # Initialize Session
    try:
        qnn_options = {'backend_path': 'QnnHtp.dll'}
        session = ort.InferenceSession(model_path, providers=[('QNNExecutionProvider', qnn_options), 'CPUExecutionProvider'])
        
        providers = session.get_providers()
        print(f"[INFO] Active Providers: {providers}")
        if 'QNNExecutionProvider' in providers:
            print(f"\033[92m[SUCCESS] QNN HTP Backend Initialized! NPU is READY.\033[0m")
        else:
            print(f"\033[93m[WARNING] QNN not found. Falling back to {providers[0]}.\033[0m")
            
    except Exception as e:
        print(f"\033[91m[ERROR] Failed to initialize QNN: {e}\033[0m")
        return

    # Data Setup
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    if isinstance(input_shape[0], str): input_shape[0] = 1
    
    image_paths = list(Path(data_dir).glob('*.jpg'))
    if len(image_paths) == 0:
        print("[ERROR] No images found. Run 'download_assets.py' first.")
        return
    
    print(f"[INFO] Loaded {len(image_paths)} images for inference loop.")
    print(f"[INFO] Input shape: {input_shape}")
    
    # Load labels
    labels = load_imagenet_labels()
    
    # Warmup
    print("[INFO] Warming up NPU...")
    dummy = np.zeros(input_shape, dtype=np.float32)
    for _ in range(10):
        session.run(None, {input_name: dummy})
        
    print(f"[INFO] Starting Classification Test for {duration} seconds...")
    print("       (Check Task Manager -> Performance -> NPU)")
    
    start_time = time.time()
    end_time = start_time + duration
    frame_count = 0
    img_idx = 0
    
    try:
        while time.time() < end_time:
            img_path = image_paths[img_idx % len(image_paths)]
            data = preprocess_imagenet(img_path, input_shape)
            
            # Inference
            outputs = session.run(None, {input_name: data})
            
            # Optional: Show predictions
            if show_predictions and frame_count % 100 == 0:
                predictions = postprocess_classification(outputs)
                print(f"\n[PREDICTION] Top-5 classes:")
                for idx, prob in predictions:
                    print(f"  {labels.get(idx, f'class_{idx}')}: {prob:.3f}")
            
            frame_count += 1
            img_idx += 1
            
            # Live Stats
            current_time = time.time()
            elapsed = current_time - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
                sys.stdout.write(f"\r[RUNNING] Time: {elapsed:.1f}s | Frames: {frame_count} | FPS: {fps:.2f}  ")
                sys.stdout.flush()
                
    except KeyboardInterrupt:
        print("\n[STOPPED] User interrupted.")
        
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    avg_latency = (total_time / frame_count) * 1000
    
    print("\n" + "="*40)
    print(f"CLASSIFICATION BENCHMARK COMPLETED")
    print(f"Total Frames : {frame_count}")
    print(f"Total Time   : {total_time:.2f} s")
    print(f"Avg Throughput: {avg_fps:.2f} FPS")
    print(f"Avg Latency   : {avg_latency:.2f} ms")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/mobilenetv2_qdq.onnx',
                        help='Path to quantized classification model')
    parser.add_argument('--data', type=str, default='datasets/coco128/images/train2017')
    parser.add_argument('--duration', type=int, default=30, help='Duration in seconds')
    parser.add_argument('--show-predictions', action='store_true', help='Show top-5 predictions periodically')
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"Model {args.model} not found!")
        print("Please run one of:")
        print("  python scripts/quantize_mobilenetv2.py")
        print("  python scripts/quantize_resnet50.py")
    else:
        run_npu_inference(args.model, args.data, args.duration, args.show_predictions)
