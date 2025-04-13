#!/bin/bash
set -e

# Build Rust project
echo "Building Rust project..."
cargo build

# Create build directory for C++
mkdir -p build
cd build

# Configure and build C++ project
echo "Building C++ project..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Get number of CPU cores for parallel build
if [ "$(uname)" = "Darwin" ]; then
    CORES=$(sysctl -n hw.ncpu)
else
    CORES=$(nproc)
fi

make -j$CORES
cd ..

# Ensure the library is in the correct location
mkdir -p .build/lib
cp build/lib/libbook_wrapper.a .build/lib/

# Copy the cpp_main binary to the root directory
echo "Copying C++ binary..."
cp build/cpp_main .

# Build Swift project
echo "Building Swift project..."
swift build -c release -Xlinker -L$(pwd)/.build/lib
cp .build/release/BookExample swift_main

echo "Build completed successfully!" 