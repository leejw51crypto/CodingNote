#!/bin/bash

# Build llama.cpp with shared library support

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/llama.cpp"

# Detect OS
OS="$(uname -s)"
case "$OS" in
    Darwin)
        echo "Building for macOS with Metal support..."
        cmake -B build -DBUILD_SHARED_LIBS=ON
        cmake --build build -j$(sysctl -n hw.ncpu)
        ;;
    Linux)
        echo "Building for Linux..."
        read -p "Enable CUDA support? (y/n): " USE_CUDA
        if [[ "$USE_CUDA" =~ ^[Yy]$ ]]; then
            echo "Building with CUDA..."
            cmake -B build -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON
        else
            echo "Building without CUDA..."
            cmake -B build -DBUILD_SHARED_LIBS=ON
        fi
        cmake --build build -j$(nproc)
        ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

# Copy libraries to top folder
echo "Copying libraries to project root..."
cd "$SCRIPT_DIR"
cp -fL llama.cpp/build/bin/*.dylib . 2>/dev/null || true
cp -fL llama.cpp/build/bin/*.so . 2>/dev/null || true
cp -fL llama.cpp/build/src/*.so . 2>/dev/null || true
cp -fL llama.cpp/build/ggml/src/*.so . 2>/dev/null || true

echo "Build complete!"
ls -lh *.dylib 2>/dev/null
ls -lh *.so 2>/dev/null
