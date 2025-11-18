#!/bin/bash

# Run interactive AI chat

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export LLAMA_LIB_PATH="$SCRIPT_DIR/llama.cpp/build/bin/libllama.dylib"

# Set DEBUG=1 to see llama.cpp output
# export DEBUG=1

luajit ai.lua
