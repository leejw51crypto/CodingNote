# Cap'n Proto RPC Quick Start

## Three Ways to Run

### 1️⃣  Orchestrated Test (Simplest)

```bash
python run_test.py
```

**Best for:** Quick testing, development
**Behavior:** Starts server → Runs client → Shows results → Stops server → Exits

---

### 2️⃣  Supervisor (Production)

**Start:**
```bash
./start.sh
```

**View Results:**
```bash
cat logs/rpc_client.log
```

**Stop:**
```bash
./stop.sh
```

**Best for:** Production deployment, continuous operation
**Behavior:** Server runs continuously, client runs once

---

### 3️⃣  Manual (Development)

**Terminal 1:**
```bash
python rpc_server.py
```

**Terminal 2:**
```bash
python rpc_client.py
```

**Best for:** Development, debugging
**Behavior:** Manual control of both processes

---

## Expected Output

All methods run these 7 tests:

```
1️⃣  Testing add(10, 5)...
   ✅ Result: 15

2️⃣  Testing subtract(20, 8)...
   ✅ Result: 12

3️⃣  Testing multiply(7, 6)...
   ✅ Result: 42

4️⃣  Testing divide(100, 4)...
   ✅ Result: 25.0

5️⃣  Testing evaluate('2 + 3 * 4')...
   ✅ Result: 14.0

6️⃣  Testing concurrent requests...
   ✅ Concurrent results: [3, 12, 5, 10.0]

7️⃣  Testing error handling (division by zero)...
   ✅ Error handling works! Server exception caught on client side
   📝 Exception: ValueError - Division by zero
```

---

## Files Created

### Core Implementation
- `calculator.capnp` - RPC schema
- `rpc_server.py` - Server code
- `rpc_client.py` - Client code with tests

### Orchestration
- `run_test.py` - All-in-one test runner
- `start.sh` - Supervisor start script
- `stop.sh` - Supervisor stop script
- `supervisord.conf` - Supervisor configuration

### Documentation
- `QUICKSTART.md` - This file
- `RPC_README.md` - Comprehensive guide
- `SUPERVISOR_USAGE.md` - Supervisor detailed docs

---

## Troubleshooting

**Port 60000 in use:**
```bash
lsof -ti :60000 | xargs kill -9
```

**Clean everything:**
```bash
./stop.sh
rm -rf logs/ supervisord.pid supervisord.log
```

**Install dependencies:**
```bash
pip install pycapnp supervisor
```

---

## What's Tested

✅ Basic RPC calls (add, subtract, multiply, divide)
✅ Expression evaluation
✅ Concurrent requests (asyncio.gather)
✅ Error propagation (exceptions from server to client)
✅ Zero-copy serialization
✅ Promise pipelining

---

## Architecture

```
┌─────────────┐                    ┌─────────────┐
│   Client    │ ───── RPC ────────▶│   Server    │
│             │   Port 60000       │             │
│ rpc_client  │◀──── Results ──────│ rpc_server  │
└─────────────┘                    └─────────────┘
       │                                  │
       └──────── Cap'n Proto ─────────────┘
            (Zero-copy messages)
```

---

**Need more details?** See `RPC_README.md` or `SUPERVISOR_USAGE.md`
