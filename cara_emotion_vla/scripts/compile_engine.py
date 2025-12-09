#!/bin/bash
# Run this on the Jetson!

# Path to the specific trtexec binary (standard location on JetPack)
TRTEXEC=/usr/src/tensorrt/bin/trtexec

# 1. Define paths
ONNX_MODEL="../models/onnx/model.onnx"
ENGINE_OUTPUT="../models/cara_emotion_fp16.engine"

# 2. Compile
# --fp16: Enables 16-bit floating point (Crucial for speed)
# --saveEngine: Where to save the binary blob
# --explicitBatch: Needed for ViT architectures
echo "Compiling TensorRT Engine... this takes a few minutes..."

$TRTEXEC --onnx=$ONNX_MODEL \
         --saveEngine=$ENGINE_OUTPUT \
         --fp16 \
         --explicitBatch

echo " Engine compiled: $ENGINE_OUTPUT"