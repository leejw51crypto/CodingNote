# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Cap'n Proto learning and experimentation repository using Python (pycapnp). Cap'n Proto is a high-performance serialization format with zero-copy deserialization capabilities.

## Running Examples

```bash
# Basic hello world with packed/unpacked serialization
python hello_capnp.py

# Zero-copy deserialization demonstration
python test_zero_copy.py
```

## Cap'n Proto Schema Files

- **person.capnp**: Defines a simple Person struct with name, email, and age fields
- Schema files are loaded automatically by pycapnp at runtime - no manual compilation needed
- Import pattern: `import person_capnp` (pycapnp automatically finds `person.capnp`)

## Key Architecture Concepts

### pycapnp Usage Patterns

**Creating and serializing messages:**
```python
person = person_capnp.Person.new_message()
person.name = "Alice"
serialized = person.to_bytes()           # Unpacked format
serialized_packed = person.to_bytes_packed()  # Compressed format
```

**Deserializing messages:**
- `from_bytes()` returns a **context manager** - use with `with` statement
- `from_bytes_packed()` returns a **reader object** directly (not a context manager)
- Both provide zero-copy access to fields

**Memory management:**
- Call `person.clear_write_flag()` before serializing the same message object multiple times
- This prevents memory leak warnings

### Zero-Copy Architecture

Cap'n Proto differs from traditional serialization (JSON, Protobuf):
- **Traditional**: Bytes → Parse → Copy to objects → Use
- **Cap'n Proto**: Bytes → Validate → Read directly via pointers (zero-copy)

When you call `from_bytes()` or `from_bytes_packed()`:
1. The data is validated but NOT copied into Python objects
2. Field access happens on-demand via pointer arithmetic
3. Only requested fields trigger Python object creation

### Packed vs Unpacked Format

- **Unpacked**: Standard Cap'n Proto format with 8-byte word alignment (typically ~72 bytes for Person)
- **Packed**: Compressed format that eliminates zero-byte runs (typically ~39 bytes, ~54% compression)
- `from_bytes_packed()` unpacks the compressed bytes into memory, then provides zero-copy access
- Packed format is ideal for network transmission; unpacked format is used in-memory

## Dependencies

- **pycapnp**: Python bindings for Cap'n Proto
  - Bundles C++ Cap'n Proto library if not installed system-wide
  - No separate compilation step required for `.capnp` schema files
- **supervisor**: Process management for production deployments (optional)

## RPC Examples

### Running the RPC System

Three ways to run the Calculator RPC example:

**1. Orchestrated Test (Single Command)**
```bash
python run_test.py
```
Starts server, runs client tests, displays results, stops server automatically.

**2. Supervisor (Production)**
```bash
./start.sh              # Start server and client
cat logs/rpc_client.log # View test results
./stop.sh               # Stop all processes
```
Server runs continuously, client executes once. All logs saved to `logs/`.

**3. Manual (Development)**
```bash
# Terminal 1
python rpc_server.py    # Server on localhost:60000

# Terminal 2
python rpc_client.py    # Client runs tests
```

### Critical Import Order

**IMPORTANT**: Must import `capnp` before any `*_capnp` modules:

```python
import capnp  # Must import capnp first
import calculator_capnp
```

This order is **required** for pycapnp's auto-generation to work. Linters may reorder imports alphabetically (putting `calculator_capnp` first), which will cause `ModuleNotFoundError`. Add `# Must import capnp first` comment to prevent auto-reordering.

### RPC Server Architecture

**Server Implementation Pattern:**
```python
class CalculatorImpl(calculator_capnp.Calculator.Server):
    # All RPC methods must be async
    async def add(self, a, b, **kwargs):
        return a + b

async def new_connection(stream):
    # Handle each client connection
    await capnp.TwoPartyServer(stream, bootstrap=CalculatorImpl()).on_disconnect()

# Start server
server = await capnp.AsyncIoStream.create_server(new_connection, "localhost", 60000)
```

**Key Points:**
- All interface methods must be `async`
- Use `new_connection()` handler for per-client bootstrap
- Call `.on_disconnect()` to keep connection alive
- Must use `capnp.run()` wrapped in `asyncio.run()` for event loop

### RPC Client Pattern

```python
# Connect to server
client = await capnp.AsyncIoStream.create_connection(host="localhost", port=60000)
calculator = capnp.TwoPartyClient(client).bootstrap().cast_as(calculator_capnp.Calculator)

# Make RPC calls
result = await calculator.add(10, 5)
print(result.result)  # Access returned value

# Concurrent requests
results = await asyncio.gather(
    calculator.add(1, 2),
    calculator.multiply(3, 4)
)
```

### Supervisor Process Management

**Configuration** (`supervisord.conf`):
- Server starts first (priority 10, startsecs 2)
- Client starts after server (priority 20)
- Both set to `autorestart=false` (one-shot execution)

**Status Checking:**
```bash
supervisorctl -c supervisord.conf status
```

**Log Locations:**
- `logs/rpc_server.log` - Server stdout
- `logs/rpc_client.log` - Client test results
- `logs/*_error.log` - Error output

### Orchestration Pattern (run_test.py)

The orchestrated test demonstrates a complete lifecycle:
1. Start server as background asyncio task
2. Wait 2 seconds for server initialization
3. Run client tests sequentially
4. Cancel server task on completion
5. All in single Python process with automatic cleanup
