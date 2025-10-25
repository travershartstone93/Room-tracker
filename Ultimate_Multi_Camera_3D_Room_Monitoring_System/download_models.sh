#!/bin/bash
set -e

MODELS_DIR="models"
mkdir -p $MODELS_DIR

echo "Downloading ONNX models..."

# YOLOv8x (mapping)
if [ ! -f "$MODELS_DIR/yolov8x.onnx" ]; then
    echo "Downloading YOLOv8x..."
    wget -O $MODELS_DIR/yolov8x.onnx \
        https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.onnx
fi

# YOLOv8n (monitoring detection)
if [ ! -f "$MODELS_DIR/yolov8n-int8.onnx" ]; then
    echo "Downloading YOLOv8n..."
    wget -O $MODELS_DIR/yolov8n.onnx \
        https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx
    echo "Quantizing to INT8..."
    python3 - <<EOF
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = "$MODELS_DIR/yolov8n.onnx"
model_quant = "$MODELS_DIR/yolov8n-int8.onnx"
quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
EOF
fi

# YOLOv8n-pose (pose estimation)
if [ ! -f "$MODELS_DIR/yolov8n-pose-int8.onnx" ]; then
    echo "Downloading YOLOv8n-pose..."
    wget -O $MODELS_DIR/yolov8n-pose.onnx \
        https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.onnx
    echo "Quantizing to INT8..."
    python3 - <<EOF
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = "$MODELS_DIR/yolov8n-pose.onnx"
model_quant = "$MODELS_DIR/yolov8n-pose-int8.onnx"
quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
EOF
fi

# MiDaS small (depth estimation)
if [ ! -f "$MODELS_DIR/midas_small.onnx" ]; then
    echo "Downloading MiDaS small..."
    wget -O $MODELS_DIR/midas_small.onnx \
        https://github.com/isl-org/MiDaS/releases/download/v3_1/midas_v21_small_256.pt
    # Convert PyTorch to ONNX (requires additional setup)
    echo "Note: Convert MiDaS PyTorch model to ONNX manually"
fi

# Depth-Anything v2 (heavy depth estimation)
if [ ! -f "$MODELS_DIR/depth_anything_v2.onnx" ]; then
    echo "Downloading Depth-Anything v2..."
    wget -O $MODELS_DIR/depth_anything_v2.onnx \
        https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2s.pth
    # Convert PyTorch to ONNX (requires additional setup)
    echo "Note: Convert Depth-Anything PyTorch model to ONNX manually"
fi

echo "Model download complete!"
