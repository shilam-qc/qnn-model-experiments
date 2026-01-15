@echo off
REM Multi-Model NPU Inference Launcher
REM For Snapdragon X Elite devices

echo ========================================
echo   Multi-Model NPU Inference Launcher
echo   Snapdragon X Elite
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.10+ ARM64 version
    pause
    exit /b 1
)

echo [INFO] Python detected
echo.

REM Check if dependencies are installed
echo [INFO] Checking dependencies...
pip show onnxruntime-qnn >nul 2>&1
if errorlevel 1 (
    echo [SETUP] Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo [OK] Dependencies already installed
)
echo.

REM Check if calibration data exists
if not exist "datasets\coco128\images\train2017" (
    echo [SETUP] Downloading calibration data...
    python download_assets.py
    if errorlevel 1 (
        echo [ERROR] Failed to download calibration data
        pause
        exit /b 1
    )
)

:menu
cls
echo ========================================
echo   Multi-Model NPU Inference Menu
echo ========================================
echo.
echo Select a model to run:
echo.
echo   1. Object Detection (YOLOv8)
echo   2. Pose Detection (YOLOv8-Pose)
echo   3. Classification (MobileNetV2)
echo   4. Classification (ResNet50)
echo   5. Segmentation (YOLOv8-Seg)
echo.
echo   Q. Quantize Models (First Time Setup)
echo   0. Exit
echo.
set /p choice="Enter your choice: "

if "%choice%"=="1" goto detection
if "%choice%"=="2" goto pose
if "%choice%"=="3" goto classification_mobile
if "%choice%"=="4" goto classification_resnet
if "%choice%"=="5" goto segmentation
if "%choice%"=="Q" goto quantize
if "%choice%"=="q" goto quantize
if "%choice%"=="0" goto end
echo Invalid choice. Please try again.
timeout /t 2 >nul
goto menu

:detection
cls
echo Running Object Detection (YOLOv8)...
echo.
python unified_inference.py --task detection --duration 30
echo.
pause
goto menu

:pose
cls
echo Running Pose Detection (YOLOv8-Pose)...
echo.
if not exist "models\yolov8s-pose_qdq.onnx" (
    echo [WARNING] Model not found. Quantizing first...
    python scripts\quantize_yolov8_pose.py
    if errorlevel 1 (
        echo [ERROR] Quantization failed
        pause
        goto menu
    )
)
python unified_inference.py --task pose --duration 30
echo.
pause
goto menu

:classification_mobile
cls
echo Running Classification (MobileNetV2)...
echo.
if not exist "models\mobilenetv2_qdq.onnx" (
    echo [WARNING] Model not found. Quantizing first...
    python scripts\quantize_mobilenetv2.py
    if errorlevel 1 (
        echo [ERROR] Quantization failed
        pause
        goto menu
    )
)
python unified_inference.py --task classification --duration 30
echo.
pause
goto menu

:classification_resnet
cls
echo Running Classification (ResNet50)...
echo.
if not exist "models\resnet50_qdq.onnx" (
    echo [WARNING] Model not found. Quantizing first...
    python scripts\quantize_resnet50.py
    if errorlevel 1 (
        echo [ERROR] Quantization failed
        pause
        goto menu
    )
)
python unified_inference.py --task classification --model models\resnet50_qdq.onnx --duration 30
echo.
pause
goto menu

:segmentation
cls
echo Running Segmentation (YOLOv8-Seg)...
echo.
if not exist "models\yolov8s-seg_qdq.onnx" (
    echo [WARNING] Model not found. Quantizing first...
    python scripts\quantize_yolov8_seg.py
    if errorlevel 1 (
        echo [ERROR] Quantization failed
        pause
        goto menu
    )
)
python unified_inference.py --task segmentation --duration 30
echo.
pause
goto menu

:quantize
cls
echo ========================================
echo   Model Quantization Menu
echo ========================================
echo.
echo Select models to quantize:
echo.
echo   1. YOLOv8-Pose
echo   2. MobileNetV2
echo   3. ResNet50
echo   4. YOLOv8-Seg
echo   5. All Models
echo   0. Back to Main Menu
echo.
set /p qchoice="Enter your choice: "

if "%qchoice%"=="1" (
    python scripts\quantize_yolov8_pose.py
    pause
    goto quantize
)
if "%qchoice%"=="2" (
    python scripts\quantize_mobilenetv2.py
    pause
    goto quantize
)
if "%qchoice%"=="3" (
    python scripts\quantize_resnet50.py
    pause
    goto quantize
)
if "%qchoice%"=="4" (
    python scripts\quantize_yolov8_seg.py
    pause
    goto quantize
)
if "%qchoice%"=="5" (
    echo Quantizing all models...
    python scripts\quantize_yolov8_pose.py
    python scripts\quantize_mobilenetv2.py
    python scripts\quantize_resnet50.py
    python scripts\quantize_yolov8_seg.py
    pause
    goto quantize
)
if "%qchoice%"=="0" goto menu
echo Invalid choice.
timeout /t 2 >nul
goto quantize

:end
echo.
echo Thank you for using Multi-Model NPU Inference!
echo.
exit /b 0
