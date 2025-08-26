#!/usr/bin/env bash
# =============================================================================
# ONNX MODEL EXPORT SCRIPT FOR JAVASCRIPT
# =============================================================================
# This script converts a trained PyTorch model to ONNX format for use in
# JavaScript/web browsers. ONNX (Open Neural Network Exchange) is a format
# that allows models trained in Python to run efficiently in JavaScript.
#
# Usage:   ./export_js.sh <variant> <tag>
# Example: ./export_js.sh byte shakespeare
#          ./export_js.sh gpt2 shiffman-transcripts
# =============================================================================

# Exit on any error, undefined variables, or pipe failures
set -euo pipefail

# Parse required command line arguments
VARIANT="${1:?byte or gpt2}"                           # Model variant (byte or gpt2)
TAG="${2:?dataset tag, e.g. shakespeare}" # Dataset tag for identification

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # Directory containing this script
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"                    # Project root directory

SRC_DIR="baby-${VARIANT}-${TAG}"           # Source: trained model directory
DEST_DIR="node-js/model-${VARIANT}-${TAG}" # Destination: JavaScript-ready model directory

SRC_PATH="$ROOT_DIR/$SRC_DIR"   # Full path to source model
DEST_PATH="$ROOT_DIR/$DEST_DIR" # Full path to destination directory

# Check if optimum-cli is installed (needed for ONNX conversion)
# optimum is Hugging Face's library for model optimization and conversion
command -v optimum-cli >/dev/null 2>&1 || { echo "optimum-cli not found. pip install optimum"; exit 1; }

# Clean up any existing destination directory
rm -rf "$DEST_PATH"
# Create destination directory with onnx subdirectory
mkdir -p "$DEST_PATH/onnx"

# Convert the PyTorch model to ONNX format using Hugging Face Optimum
# text-generation-with-past enables key-value caching for faster inference
optimum-cli export onnx --model "$SRC_PATH" --task text-generation-with-past "$DEST_PATH/onnx"

# Copy essential configuration and tokenizer files to destination
# These files are needed for JavaScript to understand the model and tokenizer
cp "$SRC_PATH/"{config.json,tokenizer.json,tokenizer_config.json,special_tokens_map.json,merges.txt,vocab.json} "$DEST_PATH/" 2>/dev/null || true

# Clean up the ONNX directory - keep only the essential model.onnx file
find "$DEST_PATH/onnx" -type f ! -name 'model.onnx' -delete

# Print success message
echo "Successfully exported $SRC_DIR -> $DEST_DIR"
echo "The model is now ready for use in JavaScript with Transformers.js!"
