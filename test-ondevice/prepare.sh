#!/bin/bash

# Build llama.cpp with shared library support

cd llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=ON
cmake --build build -j$(sysctl -n hw.ncpu)
