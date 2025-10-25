#!/bin/bash
# Stop Cap'n Proto RPC Server and Client

set -e

echo "🛑 Stopping Cap'n Proto RPC System..."

if [ -f supervisord.pid ]; then
    # Show status before stopping
    echo ""
    echo "📊 Current Status:"
    supervisorctl -c supervisord.conf status || true

    echo ""
    echo "🔻 Shutting down supervisor..."
    supervisorctl -c supervisord.conf shutdown || true

    # Wait for clean shutdown
    sleep 2

    # Clean up PID file
    rm -f supervisord.pid

    # Kill any remaining processes on port 60000
    lsof -ti :60000 | xargs kill -9 2>/dev/null || true

    echo "✅ RPC System stopped successfully"
else
    echo "ℹ️  Supervisor is not running"

    # Still try to clean up port
    if lsof -ti :60000 > /dev/null 2>&1; then
        echo "🔍 Found process on port 60000, cleaning up..."
        lsof -ti :60000 | xargs kill -9 2>/dev/null || true
        echo "✅ Port 60000 cleaned up"
    fi
fi
