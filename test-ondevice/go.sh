#!/bin/bash

# Run llama.lua example with Gemma 2 9B model

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export LLAMA_LIB_PATH="$SCRIPT_DIR/llama.cpp/build/bin/libllama.dylib"
export MODEL_PATH="$SCRIPT_DIR/models/gemma-2-9b-q4.gguf"

luajit example_llama.lua
