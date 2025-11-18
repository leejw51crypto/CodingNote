#!/bin/bash

# Run Love2D AI Chat GUI

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export LLAMA_LIB_PATH="$SCRIPT_DIR/llama.cpp/build/bin/libllama.dylib"

love .
