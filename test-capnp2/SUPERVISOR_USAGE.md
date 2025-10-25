# Supervisor Usage Guide

This guide explains how to use the supervisor-based orchestration for the Cap'n Proto RPC system.

## Overview

The supervisor setup automatically:
- Starts the RPC server first
- Waits 2 seconds for server initialization
- Starts the RPC client (which runs tests and exits)
- Manages all processes
- Handles clean shutdown

## Prerequisites

Install supervisor:
```bash
pip install supervisor
```

## Usage

### Starting the System

```bash
./start.sh
```

**What happens:**
1. Creates `logs/` directory if needed
2. Cleans up any existing processes
3. Starts `supervisord` with the configuration
4. Server starts and listens on port 60000
5. Client connects, runs tests, and exits
6. Server continues running

**Output:**
```
🚀 Starting Cap'n Proto RPC System...
📋 Cleaning up old supervisor processes...
🔍 Checking port 60000...
▶️  Starting supervisord...

📊 Process Status:
rpc_client    EXITED    Oct 25 05:01 PM
rpc_server    RUNNING   pid 12345, uptime 0:00:03

✅ RPC System started!

📝 View logs:
   Server: tail -f logs/rpc_server.log
   Client: tail -f logs/rpc_client.log

🛑 Stop with: ./stop.sh
```

### Viewing Logs

**Client output (test results):**
```bash
cat logs/rpc_client.log
```

**Server output:**
```bash
cat logs/rpc_server.log
```

**Real-time monitoring:**
```bash
tail -f logs/rpc_server.log
```

### Checking Status

```bash
supervisorctl -c supervisord.conf status
```

**Example output:**
```
rpc_client   EXITED    Oct 25 05:01 PM
rpc_server   RUNNING   pid 12345, uptime 0:01:30
```

### Stopping the System

```bash
./stop.sh
```

**What happens:**
1. Shows current status
2. Gracefully shuts down all processes
3. Cleans up PID file
4. Ensures port 60000 is freed

**Output:**
```
🛑 Stopping Cap'n Proto RPC System...

📊 Current Status:
rpc_client   EXITED    Oct 25 05:01 PM
rpc_server   RUNNING   pid 12345, uptime 0:02:15

🔻 Shutting down supervisor...
Shut down
✅ RPC System stopped successfully
```

## Supervisor Configuration

The `supervisord.conf` file defines:

### Process: rpc_server
- **Priority:** 10 (starts first)
- **Autostart:** true
- **Autorestart:** false (one-shot server)
- **Startsecs:** 2 (wait time before marking as started)

### Process: rpc_client
- **Priority:** 20 (starts after server)
- **Autostart:** true
- **Autorestart:** false (runs once and exits)
- **Depends on:** rpc_server

## File Structure

```
test-capnp/
├── supervisord.conf      # Supervisor configuration
├── supervisord.pid       # Process ID file (auto-created)
├── supervisord.log       # Supervisor main log (auto-created)
├── start.sh              # Start script
├── stop.sh               # Stop script
├── logs/                 # Log directory (auto-created)
│   ├── rpc_server.log        # Server stdout
│   ├── rpc_server_error.log  # Server stderr
│   ├── rpc_client.log        # Client stdout (test results)
│   └── rpc_client_error.log  # Client stderr
├── rpc_server.py         # Server implementation
├── rpc_client.py         # Client implementation
└── calculator.capnp      # RPC schema
```

## Common Operations

### Restart Just the Client

```bash
supervisorctl -c supervisord.conf start rpc_client
```

### Manually Stop Server

```bash
supervisorctl -c supervisord.conf stop rpc_server
```

### View All Logs

```bash
ls -la logs/
cat logs/*.log
```

### Clean Everything

```bash
./stop.sh
rm -rf logs/
rm -f supervisord.pid supervisord.log
```

## Troubleshooting

### Port Already in Use

```bash
lsof -ti :60000 | xargs kill -9
./start.sh
```

### Supervisor Won't Start

```bash
rm -f supervisord.pid /tmp/supervisor.sock
./start.sh
```

### View Detailed Status

```bash
supervisorctl -c supervisord.conf status
cat supervisord.log
```

### Debug Mode

Edit `supervisord.conf` and change:
```ini
[supervisord]
loglevel=debug
```

Then restart:
```bash
./stop.sh
./start.sh
```

## Benefits of Supervisor

✅ **Automatic startup ordering** - Server starts before client
✅ **Process monitoring** - Track process state
✅ **Centralized logging** - All logs in one place
✅ **Graceful shutdown** - Clean process termination
✅ **Production ready** - Suitable for deployment
✅ **Easy management** - Simple start/stop scripts

## Alternative: Orchestrated Test

For simple testing without supervisor:

```bash
python run_test.py
```

This runs everything in a single Python process with automatic cleanup.
