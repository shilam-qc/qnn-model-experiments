#!/usr/bin/env python3
"""
Unified Multi-Model NPU Inference
Supports: Object Detection, Pose Detection, Classification, Segmentation
"""
import argparse
import sys
from pathlib import Path

# Import inference modules
sys.path.append(str(Path(__file__).parent))

def print_main_banner():
    print(r"""
   __  ___      ____  _      __  ___         __     __
  /  |/  /_ __/ / /_(_)____/  |/  /__  ____/ /__  / /
 / /|_/ / // / / __/ /___/ /|_/ / _ \/ __  / -_) / / 
/_/  /_/\_,_/_/\__/_/   /_/  /_/\___/\_,_/\__/  /_/  
    NPU INFERENCE | SNAPDRAGON X ELITE
    """)
    print("=" * 60)
    print("Available Models:")
    print("  1. Object Detection  (YOLOv8)")
    print("  2. Pose Detection    (YOLOv8-Pose)")
    print("  3. Classification    (MobileNetV2 / ResNet50)")
    print("  4. Segmentation      (YOLOv8-Seg)")
    print("=" * 60)

def run_detection(args):
    """Run object detection inference"""
    from npu_realtime_inference import run_npu_inference
    model_path = args.model or 'yolov8s_qdq.onnx'
    print(f"\n🎯 Running Object Detection with {model_path}")
    run_npu_inference(model_path, args.data, args.duration)

def run_pose(args):
    """Run pose detection inference"""
    sys.path.insert(0, str(Path(__file__).parent / 'inference'))
    from npu_pose_inference import run_npu_inference
    model_path = args.model or 'models/yolov8s-pose_qdq.onnx'
    print(f"\n🧍 Running Pose Detection with {model_path}")
    run_npu_inference(model_path, args.data, args.duration)

def run_classification(args):
    """Run classification inference"""
    sys.path.insert(0, str(Path(__file__).parent / 'inference'))
    from npu_classification_inference import run_npu_inference
    model_path = args.model or 'models/mobilenetv2_qdq.onnx'
    print(f"\n🏷️  Running Classification with {model_path}")
    run_npu_inference(model_path, args.data, args.duration, args.show_predictions)

def run_segmentation(args):
    """Run segmentation inference"""
    sys.path.insert(0, str(Path(__file__).parent / 'inference'))
    from npu_segmentation_inference import run_npu_inference
    model_path = args.model or 'models/yolov8s-seg_qdq.onnx'
    print(f"\n✂️  Running Segmentation with {model_path}")
    run_npu_inference(model_path, args.data, args.duration)

def main():
    print_main_banner()
    
    parser = argparse.ArgumentParser(
        description='Unified NPU Inference for Multiple AI Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Object Detection (YOLOv8)
  python unified_inference.py --task detection
  
  # Pose Detection (YOLOv8-Pose)
  python unified_inference.py --task pose
  
  # Classification (MobileNetV2)
  python unified_inference.py --task classification
  
  # Classification (ResNet50)
  python unified_inference.py --task classification --model models/resnet50_qdq.onnx
  
  # Segmentation (YOLOv8-Seg)
  python unified_inference.py --task segmentation
        """
    )
    
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=['detection', 'pose', 'classification', 'segmentation'],
        help='Task type to run'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to quantized ONNX model (optional, uses default for task)'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='datasets/coco128/images/train2017',
        help='Path to image directory'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=30,
        help='Duration in seconds for inference test'
    )
    
    parser.add_argument(
        '--show-predictions',
        action='store_true',
        help='Show predictions during inference (classification only)'
    )
    
    args = parser.parse_args()
    
    # Route to appropriate inference function
    task_map = {
        'detection': run_detection,
        'pose': run_pose,
        'classification': run_classification,
        'segmentation': run_segmentation
    }
    
    try:
        task_map[args.task](args)
    except KeyboardInterrupt:
        print("\n\n[STOPPED] User interrupted.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
