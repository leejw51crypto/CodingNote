# Cap'n Proto Multi-Language Research Project

This project demonstrates Cap'n Proto serialization and RPC functionality across multiple programming languages including C++, Swift, Node.js, and Rust.

## Project Overview

Cap'n Proto is a fast data interchange format and RPC system. This research project explores its implementation and usage across different programming languages, demonstrating interoperability and performance characteristics.

## Project Structure

```
.
├── build.sh       # Build script for C++ and Swift implementations
├── nodejs.sh      # Build and run script for Node.js implementation
└── src/           # Source code directory for all implementations
    ├── cpp/       # C++ implementation
    ├── swift/     # Swift implementation
    ├── nodejs/    # Node.js implementation
    └── rust/      # Rust implementation
```

## Build Instructions

### C++ and Swift
```bash
./build.sh
```

### Node.js
```bash
./nodejs.sh
```

### Rust
```bash
cargo run
```

## Requirements

- Cap'n Proto compiler (capnp)
- C++ compiler (for C++ implementation)
- Swift compiler (for Swift implementation)
- Node.js and npm (for Node.js implementation)
- Rust and Cargo (for Rust implementation)

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing

Feel free to contribute to this research project by submitting pull requests or opening issues for discussion.
