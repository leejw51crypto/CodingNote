#!/bin/bash
# Start Cap'n Proto RPC Server and Client using Supervisor

set -e

echo "🚀 Starting Cap'n Proto RPC System..."

# Create logs directory if it doesn't exist
mkdir -p logs

# Check if supervisor is installed
if ! command -v supervisord &> /dev/null; then
    echo "❌ Error: supervisord is not installed"
    echo "Install with: pip install supervisor"
    exit 1
fi

# Kill any existing supervisor processes
if [ -f supervisord.pid ]; then
    echo "📋 Cleaning up old supervisor processes..."
    supervisorctl -c supervisord.conf shutdown 2>/dev/null || true
    sleep 1
    rm -f supervisord.pid
fi

# Kill any processes using port 60000
echo "🔍 Checking port 60000..."
lsof -ti :60000 | xargs kill -9 2>/dev/null || true

# Start supervisor
echo "▶️  Starting supervisord..."
supervisord -c supervisord.conf

# Wait for supervisor to start
sleep 2

# Show status
echo ""
echo "📊 Process Status:"
supervisorctl -c supervisord.conf status

echo ""
echo "✅ RPC System started!"
echo ""
echo "📝 View logs:"
echo "   Server: tail -f logs/rpc_server.log"
echo "   Client: tail -f logs/rpc_client.log"
echo ""
echo "🛑 Stop with: ./stop.sh"
