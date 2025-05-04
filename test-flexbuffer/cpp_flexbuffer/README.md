# C++ FlexBuffer Example

This is a C++ implementation of a FlexBuffer-like serialization example, based on the original Rust implementation. This example demonstrates the core concepts of FlexBuffers, with a simplified implementation of the FlexBuffers API.

## Overview

FlexBuffers is a schema-less binary serialization format that allows for efficient storage and retrieval of hierarchical data. This example demonstrates the following concepts:

1. Basic serialization and deserialization of structured data
2. Working with nested structures
3. Manual construction of serialized data
4. Schema evolution (forward and backward compatibility)
5. Binary data handling
6. Data manipulation (adding/removing fields)

## Requirements

- C++17 compatible compiler
- CMake 3.10 or higher (for building)

## How to Build and Run

```bash
# Create build directory
mkdir -p build
cd build

# Configure and build
cmake ..
make

# Run the example
cd ..
./flexbuffer_example
```

## Implementation Notes

- This example provides a simplified version of the FlexBuffers API
- The `flexbuffers.h` header provides a mock implementation of the FlexBuffers library
- In a real application, you would use the actual FlexBuffers library from FlatBuffers
- The example demonstrates the same serialization concepts as the Rust version
- C++17 features like `std::optional` are used to match the Rust functionality

## Key Components

- Data models (`User`, `Address`, `UserWithAddress`, etc.)
- FlexBuffer mock implementation (`flexbuffers.h`)
- Schema evolution demonstration with `ProductV1` and `ProductV2`
- Examples of various serialization scenarios 