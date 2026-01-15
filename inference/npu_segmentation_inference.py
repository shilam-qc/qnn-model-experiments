import time
import argparse
import numpy as np
import onnxruntime as ort
import cv2
from pathlib import Path
import sys

def print_banner():
    print(r"""
   ____                             __      __  _          
  / __/__ ___ ___ _  ___ ___  ___ _/ /___ _/ /_(_)__  ___  
 _\ \/ -_) _ `/  ' \/ -_) _ \/ _ `/ __/ _ `/ __/ / _ \/ _ \ 
/___/\__/\_, /_/_/_/\__/_//_/\_,_/\__/\_,_/\__/_/\___/_//_/ 
        /___/                                                
     ON SNAPDRAGON X ELITE | HEXAGON NPU
    """)

def preprocess(image_path, input_shape):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = input_shape[2], input_shape[3]
    img = cv2.resize(img, (w, h))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

def postprocess_segmentation(outputs, conf_threshold=0.5):
    """
    YOLOv8-Seg output format:
    - Detection boxes + segmentation masks
    - Typically 2 outputs: boxes and masks
    """
    # This is a simplified version - actual postprocessing depends on exact output format
    detections = []
    if len(outputs) >= 2:
        boxes_output = outputs[0]  # Detection boxes
        masks_output = outputs[1]  # Segmentation masks
        
        # Process detections (simplified)
        num_detections = min(10, boxes_output.shape[-1] if len(boxes_output.shape) > 1 else 1)
        for i in range(num_detections):
            detections.append({
                'bbox': [0, 0, 100, 100],  # Placeholder
                'confidence': 0.9,
                'mask_shape': masks_output.shape if len(outputs) > 1 else None
            })
    return detections

def run_npu_inference(model_path, data_dir, duration=20):
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
    
    # Print output info
    print(f"[INFO] Model outputs: {len(session.get_outputs())}")
    for i, output in enumerate(session.get_outputs()):
        print(f"  Output {i}: {output.name}, shape: {output.shape}")
    
    image_paths = list(Path(data_dir).glob('*.jpg'))
    if len(image_paths) == 0:
        print("[ERROR] No images found. Run 'download_assets.py' first.")
        return
    
    print(f"[INFO] Loaded {len(image_paths)} images for inference loop.")
    
    # Warmup
    print("[INFO] Warming up NPU...")
    dummy = np.zeros(input_shape, dtype=np.float32)
    for _ in range(10):
        session.run(None, {input_name: dummy})
        
    print(f"[INFO] Starting Segmentation Test for {duration} seconds...")
    print("       (Check Task Manager -> Performance -> NPU)")
    
    start_time = time.time()
    end_time = start_time + duration
    frame_count = 0
    img_idx = 0
    
    try:
        while time.time() < end_time:
            img_path = image_paths[img_idx % len(image_paths)]
            data = preprocess(img_path, input_shape)
            
            # Inference
            outputs = session.run(None, {input_name: data})
            
            # Optional: Process segmentation masks
            # segments = postprocess_segmentation(outputs)
            
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
    print(f"SEGMENTATION BENCHMARK COMPLETED")
    print(f"Total Frames : {frame_count}")
    print(f"Total Time   : {total_time:.2f} s")
    print(f"Avg Throughput: {avg_fps:.2f} FPS")
    print(f"Avg Latency   : {avg_latency:.2f} ms")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/yolov8s-seg_qdq.onnx')
    parser.add_argument('--data', type=str, default='datasets/coco128/images/train2017')
    parser.add_argument('--duration', type=int, default=30, help='Duration in seconds')
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"Model {args.model} not found! Please run 'python scripts/quantize_yolov8_seg.py' first.")
    else:
        run_npu_inference(args.model, args.data, args.duration)
