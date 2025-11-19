#!/bin/bash

# Run llama.lua example with Gemma 3 1B model

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

luajit example_llama.lua
