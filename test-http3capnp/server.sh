#!/bin/bash
# Make sure uvicorn is installed
pip install uvicorn fastapi >/dev/null 2>&1
# Start the HTTP/3 server
python server.py --host 0.0.0.0 --http1-port 8443 --http3-port 4433 --cert ./certs/server.crt --key ./certs/server.key --verbose