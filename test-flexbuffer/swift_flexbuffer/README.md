# Swift FlexBuffer Example

This is a Swift implementation of a FlexBuffer-like serialization example, based on the original Rust implementation. Since Swift doesn't have a native FlexBuffer library, this example uses JSON serialization to demonstrate similar concepts.

## Overview

FlexBuffers is a schema-less binary serialization format that allows for efficient storage and retrieval of hierarchical data. This example demonstrates the following concepts using Swift's JSON serialization as a stand-in:

1. Basic serialization and deserialization of structured data
2. Working with nested structures
3. Manual construction of serialized data
4. Schema evolution (forward and backward compatibility)
5. Binary data handling
6. Data manipulation (adding/removing fields)

## How to Run

To run this example, you'll need Xcode or Swift installed on your system.

```bash
swift FlexBufferExample.swift
```

## Implementation Notes

- This example uses Swift's `Codable` protocol for serialization/deserialization
- `JSONEncoder` and `JSONDecoder` are used to simulate FlexBuffer operations
- Custom `Builder` and `Reader` classes emulate FlexBuffer's construction and reading capabilities
- The schema evolution example demonstrates how to handle data format changes while maintaining compatibility

## Differences from Rust Version

- Uses Swift's type system and syntax
- Relies on JSON instead of actual FlexBuffer format
- Implements Swift-idiomatic random data generation
- Uses Swift's error handling mechanisms

## Key Components

- Data models (`User`, `Address`, `UserWithAddress`, etc.)
- FlexBuffer simulation classes (`FlexBuffers`, `Builder`, `Reader`)
- Schema evolution demonstration with `ProductV1` and `ProductV2`
- Examples of different serialization scenarios 