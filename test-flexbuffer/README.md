# FlexBuffer Examples

This repository contains examples of using FlexBuffers serialization format in multiple programming languages:

- Rust (original implementation)
- Swift
- Python
- C++

## What is FlexBuffers?

FlexBuffers is a schema-less binary serialization format that is part of the Google FlatBuffers project. It allows for efficient storage and retrieval of hierarchical data without requiring a schema definition upfront.

Key features of FlexBuffers:
- Schema-less design
- Backward and forward compatibility
- Lazy parsing
- Direct access to nested data
- Compact binary representation

## Examples Overview

Each language implementation demonstrates the following concepts:
1. Basic serialization and deserialization
2. Working with nested structures
3. Manual construction of serialized data
4. Schema evolution (forward and backward compatibility)
5. Binary data handling
6. Data manipulation (adding/removing fields)

## Languages

### Rust

The original implementation using the actual FlexBuffers library.

- Location: `src/main.rs`
- Run with: `cargo run` or `make rust`

### Swift 

An implementation simulating FlexBuffers concepts using JSON serialization.

- Location: `swift_flexbuffer/FlexBufferExample.swift`
- Run with: `cd swift_flexbuffer && swift FlexBufferExample.swift` or `make swift`

### Python

An implementation simulating FlexBuffers concepts using JSON serialization.

- Location: `python_flexbuffer/flexbuffer_example.py`  
- Run with: `cd python_flexbuffer && python3 flexbuffer_example.py` or `make python`

### C++

An implementation using a simplified mock of the FlexBuffers API.

- Location: `cpp_flexbuffer/flexbuffer_example.cpp`
- Build and run with: `cd cpp_flexbuffer && mkdir -p build && cd build && cmake .. && make && cd .. && ./flexbuffer_example` or `make cpp`

## Running All Examples

Use the Makefile to run all examples:

```bash
make
```

## Implementation Notes

- The Rust implementation uses the actual FlexBuffers library.
- The Swift, Python, and C++ versions simulate the FlexBuffers functionality:
  - Swift and Python use JSON serialization as a stand-in
  - C++ provides a simplified mock of the FlexBuffers API

## Requirements

- Rust toolchain
- Swift 5.0+
- Python 3.7+
- C++17 compatible compiler and CMake 3.10+ 