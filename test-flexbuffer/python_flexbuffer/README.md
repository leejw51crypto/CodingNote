# Python FlexBuffer Example

This is a Python implementation of a FlexBuffer-like serialization example, based on the original Rust implementation. Since Python doesn't have a native FlexBuffer library, this example uses JSON serialization to demonstrate similar concepts.

## Overview

FlexBuffers is a schema-less binary serialization format that allows for efficient storage and retrieval of hierarchical data. This example demonstrates the following concepts using Python's JSON serialization as a stand-in:

1. Basic serialization and deserialization of structured data
2. Working with nested structures
3. Manual construction of serialized data
4. Schema evolution (forward and backward compatibility)
5. Binary data handling
6. Data manipulation (adding/removing fields)

## Requirements

- Python 3.7+

## How to Run

```bash
python flexbuffer_example.py
```

## Implementation Notes

- This example uses Python's `dataclasses` for structured data
- JSON serialization via the `json` module simulates FlexBuffer operations
- Custom `Builder` and `Reader` classes emulate FlexBuffer's construction and reading capabilities
- The schema evolution example demonstrates how to handle data format changes while maintaining compatibility
- Binary data is handled using base64 encoding

## Differences from Rust Version

- Uses Python's type system and dataclasses
- Relies on JSON instead of actual FlexBuffer format
- Takes advantage of Python's dynamic typing for simpler implementations
- Uses Python's standard library modules for serialization

## Key Components

- Data models (`User`, `Address`, `UserWithAddress`, etc.)
- FlexBuffer simulation classes (`FlexBuffers`, `Builder`, `Reader`)
- Schema evolution demonstration with `ProductV1` and `ProductV2`
- Examples of different serialization scenarios 