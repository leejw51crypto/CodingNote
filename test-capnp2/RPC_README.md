# Cap'n Proto RPC Example

This directory contains a complete Cap'n Proto RPC example implementing a Calculator service with multiple ways to run it.

## Files

### Core RPC Implementation
- **calculator.capnp**: RPC schema defining the Calculator interface
- **rpc_server.py**: Server implementation hosting the Calculator service
- **rpc_client.py**: Client making RPC calls to the server

### Orchestration & Management
- **run_test.py**: All-in-one orchestrated test (starts server, runs client, stops server)
- **start.sh**: Start server and client using supervisor
- **stop.sh**: Stop all processes gracefully
- **supervisord.conf**: Supervisor configuration for process management

## Quick Start

### Option 1: Orchestrated Test (Recommended for Testing)

Single command runs everything automatically:

```bash
python run_test.py
```

This will:
1. Start the RPC server
2. Wait for server to be ready
3. Run all client tests
4. Display results
5. Stop the server
6. Exit cleanly

### Option 2: Supervisor (Recommended for Production)

Start both server and client as managed processes:

```bash
./start.sh
```

View the results:
```bash
cat logs/rpc_client.log
```

Stop everything:
```bash
./stop.sh
```

### Option 3: Manual (For Development)

**Terminal 1 - Start the Server:**

```bash
python rpc_server.py
```

You should see:
```
Calculator RPC server listening on localhost:60000
Press Ctrl+C to stop the server
```

**Terminal 2 - Run the Client:**

```bash
python rpc_client.py
```

The client will connect and perform various calculator operations, showing results from both client and server perspectives.

## Architecture

### RPC Schema (calculator.capnp)

Defines the Calculator interface with methods:
- `add(a, b)` - Addition
- `subtract(a, b)` - Subtraction
- `multiply(a, b)` - Multiplication
- `divide(a, b)` - Division (returns float)
- `evaluate(expression)` - Expression evaluation

### Server (rpc_server.py)

Key components:
```python
class CalculatorImpl(calculator_capnp.Calculator.Server):
    # Implement each RPC method as async function
    async def add(self, a, b, _context, **kwargs):
        return a + b
```

Server setup:
```python
server = await capnp.AsyncIoStream.create_server(
    lambda: capnp.TwoPartyServer(bootstrap=calculator),
    host, port
)
```

### Client (rpc_client.py)

Connection:
```python
client = await capnp.AsyncIoStream.create_connection(host=host, port=port)
calculator = capnp.TwoPartyClient(client).bootstrap().cast_as(calculator_capnp.Calculator)
```

Making calls:
```python
result = await calculator.add(10, 5)
print(result.result)  # Access the returned value
```

## Features Demonstrated

1. **Basic RPC calls**: Simple request-response pattern
2. **Different data types**: Int32, Float32, Text
3. **Concurrent requests**: Multiple simultaneous RPC calls using `asyncio.gather()`
4. **Error handling**: Exceptions propagated from server to client
5. **Async/await**: Full async support with asyncio

## How Cap'n Proto RPC Works

1. **Schema Definition**: Interface defined in `.capnp` file
2. **Server Implementation**:
   - Implements the interface methods
   - Uses `TwoPartyServer` to handle incoming connections
   - Each method receives parameters and returns results
3. **Client Usage**:
   - Connects to server using `TwoPartyClient`
   - Calls methods on remote object as if local
   - Results are promises that resolve asynchronously

## Key Concepts

### Two-Party Protocol
- Simple RPC pattern: one client, one server
- Built on Cap'n Proto's message passing
- Bi-directional communication over single connection

### Promise Pipelining
- Client can make dependent calls without waiting
- Cap'n Proto automatically pipelines requests
- Reduces round-trip latency

### Zero-Copy
- RPC messages use Cap'n Proto serialization
- Same zero-copy benefits as regular Cap'n Proto
- Efficient for large data transfers

## Advanced Usage

### Concurrent Requests

```python
# Make multiple calls in parallel
results = await asyncio.gather(
    calculator.add(1, 2),
    calculator.multiply(3, 4),
    calculator.subtract(10, 5)
)
```

### Error Handling

```python
try:
    result = await calculator.divide(10, 0)
except Exception as e:
    print(f"RPC call failed: {e}")
```

### Long-Running Connections

The server keeps connections alive. Clients can make multiple requests over the same connection without reconnecting.

## Comparison with Other RPC Frameworks

| Feature | Cap'n Proto | gRPC | JSON-RPC |
|---------|-------------|------|----------|
| Serialization | Zero-copy binary | Protobuf | JSON text |
| Schema | Required | Required | Optional |
| Performance | Fastest | Fast | Slower |
| Promise pipelining | Yes | Limited | No |
| Language support | Growing | Extensive | Universal |

## Troubleshooting

**"Connection refused"**: Make sure the server is running first

**"Module not found: calculator_capnp"**: The pycapnp library auto-generates this from `calculator.capnp` at import time

**Port already in use**: Change the port number in both server and client

## Next Steps

- Add more complex data structures (structs, lists)
- Implement streaming RPC methods
- Add authentication/security
- Handle connection failures and reconnection
- Implement capability-based security (Cap'n Proto's unique feature)
