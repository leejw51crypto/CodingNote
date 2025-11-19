#!/bin/bash

# Run Love2D AI Chat GUI

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Detect OS and set library path
OS="$(uname -s)"
case "$OS" in
    Darwin)
        export LLAMA_LIB_PATH="$SCRIPT_DIR/libllama.dylib"
        ;;
    Linux)
        export LLAMA_LIB_PATH="$SCRIPT_DIR/libllama.so"
        ;;
esac

export MODEL_PATH="$SCRIPT_DIR/models/gemma-3-1b-q4.gguf"

# Set DEBUG=1 to see llama.cpp output
# export DEBUG=1

love .
