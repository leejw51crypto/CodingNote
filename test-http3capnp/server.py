#!/usr/bin/env python3

import argparse
import asyncio
import logging
import os
import ssl
import signal
import sys
import json
import datetime
from typing import Dict, List, Optional

# HTTP/1.1 support
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse

# HTTP/3 support
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.asyncio.server import serve
from aioquic.h3.connection import H3_ALPN, H3Connection
from aioquic.h3.events import H3Event, HeadersReceived, DataReceived
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import QuicEvent

# Cap'n Proto support
import hellocapnp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dual-server")


# Function to configure logger based on verbosity
def configure_logger(verbose: bool):
    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    return logger


# Create FastAPI application for HTTP/1.1
app = FastAPI()


# FastAPI routes
@app.get("/")
async def homepage():
    logger.info("HTTP/1.1 request for /")
    return PlainTextResponse("Welcome to HTTP Dual Server!")


@app.get("/echo")
async def echo_get(request: Request):
    # For GET, just echo back the query parameters
    params = dict(request.query_params)
    response_data = (
        json.dumps(params)
        if params
        else "Echo endpoint. Use POST to echo request body or add query parameters to GET."
    )
    logger.info(f"HTTP/1.1 Echo GET request with params: {params}")
    return PlainTextResponse(response_data)


@app.post("/echo")
async def echo(request: Request):
    body = await request.body()
    logger.info(f"HTTP/1.1 Echo request: {body.decode()}")
    return PlainTextResponse(body.decode())


@app.get("/hello/{argument}")
async def hello(argument: str):
    logger.info(f"HTTP/1.1 Hello request with argument: {argument}")
    return PlainTextResponse(f"Hello world {argument}")


@app.get("/gettime")
async def gettime():
    logger.info("HTTP/1.1 Time request received")
    local_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    utc_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    return JSONResponse({"local": local_time, "utc": utc_time})


@app.post("/capnp/hello")
async def capnp_hello(request: Request):
    # Accept binary Cap'n Proto, decode, and respond with Cap'n Proto
    body = await request.body()
    try:
        msg = hellocapnp.decode_hello_world(body)
        logger.info(f"HTTP/1.1 Capnp POST: {msg}")
        response = hellocapnp.encode_hello_world(f"Hello, {msg} (from server)")
        return PlainTextResponse(response, media_type="application/x-capnp")
    except Exception as e:
        logger.error(f"Capnp decode error: {e}")
        return PlainTextResponse("Invalid Capnp data", status_code=400)


# HTTP/3 Handler
class HttpServerProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._http = H3Connection(self._quic)
        self._handlers = {}
        self._paths = {}
        self._buffer = {}
        logger.info(f"New HTTP/3 connection established")

    def http_event_received(self, event: H3Event) -> None:
        if isinstance(event, HeadersReceived):
            headers = {}
            path = None
            method = None
            for header, value in event.headers:
                if header == b":method":
                    method = value.decode()
                elif header == b":path":
                    path = value.decode()
                elif header == b"host":
                    headers["host"] = value.decode()
                elif header == b"user-agent":
                    headers["user-agent"] = value.decode()

            if path is not None:
                # Store path and method
                self._paths[event.stream_id] = path
                if method == "POST":
                    # For POST, we need to buffer the data
                    self._buffer[event.stream_id] = b""
                    # Set handler for this stream
                    if path.startswith("/echo"):
                        logger.info(f"HTTP/3 POST request received for '/echo'")
                        self._handlers[event.stream_id] = self.handle_echo
                    elif path == "/capnp/hello":
                        logger.info(f"HTTP/3 POST /capnp/hello")
                        self._handlers[event.stream_id] = self.handle_capnp_hello
                    else:
                        logger.info(
                            f"HTTP/3 POST request received for unknown path: {path}"
                        )
                        self._handlers[event.stream_id] = self.handle_unknown
                elif method == "GET":
                    # Process GET requests immediately
                    if path == "/":
                        logger.info(f"HTTP/3 GET request received for '/'")
                        self.send_response(
                            event.stream_id, "Welcome to HTTP Dual Server!"
                        )
                    elif path.startswith("/hello/"):
                        argument = path.split("/hello/")[1]
                        logger.info(
                            f"HTTP/3 GET request received for '/hello/{argument}'"
                        )
                        self.send_response(event.stream_id, f"Hello world {argument}")
                    elif path == "/gettime":
                        logger.info(f"HTTP/3 GET request received for '/gettime'")
                        local_time = datetime.datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        utc_time = datetime.datetime.utcnow().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        self.send_json_response(
                            event.stream_id, {"local": local_time, "utc": utc_time}
                        )
                    elif path.startswith("/echo"):
                        logger.info(f"HTTP/3 GET request received for '/echo'")
                        # Extract query parameters
                        query_params = {}
                        if "?" in path:
                            query_str = path.split("?", 1)[1]
                            for param in query_str.split("&"):
                                if "=" in param:
                                    key, value = param.split("=", 1)
                                    query_params[key] = value

                        response_data = (
                            json.dumps(query_params)
                            if query_params
                            else "Echo endpoint. Use POST to echo request body or add query parameters to GET."
                        )
                        logger.info(
                            f"HTTP/3 Echo GET request with params: {query_params}"
                        )

                        if query_params:
                            self.send_json_response(event.stream_id, query_params)
                        else:
                            self.send_response(event.stream_id, response_data)
                    elif path == "/capnp/hello":
                        logger.info(f"HTTP/3 GET /capnp/hello")
                        # Respond with a Cap'n Proto HelloWorld message
                        response = hellocapnp.encode_hello_world(
                            "Hello from HTTP/3 server!"
                        )
                        self.send_capnp_response(event.stream_id, response)
                        return
                    else:
                        logger.info(
                            f"HTTP/3 GET request received for unknown path: {path}"
                        )
                        self.send_response(
                            event.stream_id, f"No handler for {path}", status=404
                        )

        elif isinstance(event, DataReceived):
            # Check if this stream has a handler
            if event.stream_id in self._handlers:
                # Add data to buffer
                if event.stream_id in self._buffer:
                    self._buffer[event.stream_id] += event.data

                # Process the request based on accumulated data
                # In HTTP/3, we'll process the request as soon as we receive any data
                # This is a simplification; in a real-world scenario, you might want to
                # wait for a specific signal that the request is complete
                handler = self._handlers[event.stream_id]
                data = self._buffer.get(event.stream_id, b"")
                if handler == self.handle_capnp_hello:
                    handler(event.stream_id, data)
                else:
                    handler(event.stream_id, data)

                # Clean up
                self._handlers.pop(event.stream_id, None)
                self._buffer.pop(event.stream_id, None)

    def handle_echo(self, stream_id: int, data: bytes) -> None:
        path = self._paths.get(stream_id, "")
        try:
            data_str = data.decode("utf-8")
            logger.info(f"HTTP/3 Echo request data: {data_str}")
            self.send_response(stream_id, data_str)
        except UnicodeDecodeError:
            logger.warning(
                f"HTTP/3 Echo request with non-UTF-8 data: {len(data)} bytes"
            )
            self.send_response(stream_id, f"Received {len(data)} bytes of binary data")

    def handle_unknown(self, stream_id: int, data: bytes) -> None:
        path = self._paths.get(stream_id, "unknown path")
        logger.warning(f"HTTP/3 POST request to unknown path: {path}")
        self.send_response(
            stream_id, f"No handler for POST request to {path}", status=404
        )

    def handle_capnp_hello(self, stream_id: int, data: bytes) -> None:
        try:
            msg = hellocapnp.decode_hello_world(data)
            logger.info(f"HTTP/3 Capnp POST: {msg}")
            response = hellocapnp.encode_hello_world(
                f"Hello, {msg} (from HTTP/3 server)"
            )
            self.send_capnp_response(stream_id, response)
        except Exception as e:
            logger.error(f"Capnp decode error: {e}")
            self.send_response(stream_id, "Invalid Capnp data", status=400)

    def send_response(self, stream_id: int, body: str, status: int = 200) -> None:
        logger.info(
            f"HTTP/3 sending response: {body[:50]}{'...' if len(body) > 50 else ''}"
        )
        self._http.send_headers(
            stream_id=stream_id,
            headers=[
                (b":status", str(status).encode()),
                (b"content-type", b"text/plain"),
                (b"server", b"aioquic"),
            ],
        )
        self._http.send_data(stream_id=stream_id, data=body.encode(), end_stream=True)

    def send_json_response(self, stream_id: int, data: dict, status: int = 200) -> None:
        json_data = json.dumps(data).encode()
        logger.info(f"HTTP/3 sending JSON response: {len(json_data)} bytes")
        self._http.send_headers(
            stream_id=stream_id,
            headers=[
                (b":status", str(status).encode()),
                (b"content-type", b"application/json"),
                (b"server", b"aioquic"),
            ],
        )
        self._http.send_data(stream_id=stream_id, data=json_data, end_stream=True)

    def send_capnp_response(
        self, stream_id: int, data: bytes, status: int = 200
    ) -> None:
        self._http.send_headers(
            stream_id=stream_id,
            headers=[
                (b":status", str(status).encode()),
                (b"content-type", b"application/x-capnp"),
                (b"server", b"aioquic"),
            ],
        )
        self._http.send_data(stream_id=stream_id, data=data, end_stream=True)

    def quic_event_received(self, event: QuicEvent) -> None:
        for http_event in self._http.handle_event(event):
            self.http_event_received(http_event)


# Run HTTP/3 server
async def run_http3_server(
    host: str,
    port: int,
    config: QuicConfiguration,
) -> None:
    logger.info(f"Starting HTTP/3 server on {host}:{port}")

    await serve(
        host=host,
        port=port,
        configuration=config,
        create_protocol=HttpServerProtocol,
    )

    # Keep the server running
    await asyncio.Future()


# Run HTTP/1.1 server
async def run_http1_server(
    host: str,
    port: int,
    ssl_context: Optional[ssl.SSLContext] = None,
    cert_file: Optional[str] = None,
    key_file: Optional[str] = None,
) -> None:
    """Run a HTTP/1.1 server using uvicorn."""
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        ssl_certfile=cert_file,
        ssl_keyfile=key_file,
        log_level="info",
    )

    server = uvicorn.Server(config)
    logger.info(f"Starting HTTP/1.1 server on {host}:{port}")
    await server.serve()


# Run both servers
async def run_both_servers(args):
    # Configure TLS certificates
    http3_config = QuicConfiguration(
        alpn_protocols=H3_ALPN,
        is_client=False,
        max_datagram_frame_size=65536,
    )

    # Load certificates for HTTP/3
    http3_config.load_cert_chain(args.cert, args.key)

    # Create SSL context for HTTP/1.1
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(args.cert, args.key)

    # Run both servers
    http3_task = asyncio.create_task(
        run_http3_server(args.host, args.http3_port, http3_config)
    )

    http1_task = asyncio.create_task(
        run_http1_server(args.host, args.http1_port, ssl_context, args.cert, args.key)
    )

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig, lambda: asyncio.create_task(shutdown(http3_task, http1_task))
        )

    # Start background task to listen for 'q' to quit
    quit_task = asyncio.create_task(wait_for_quit(http3_task, http1_task))

    # Wait for both tasks to complete
    try:
        await asyncio.gather(http3_task, http1_task, quit_task)
    except asyncio.CancelledError:
        logger.info("Server tasks cancelled (shutdown requested).")


async def shutdown(http3_task, http1_task):
    """Graceful shutdown."""
    logger.info("Shutting down...")

    # Cancel both tasks
    http3_task.cancel()
    http1_task.cancel()

    try:
        await http3_task
    except asyncio.CancelledError:
        pass

    try:
        await http1_task
    except asyncio.CancelledError:
        pass

    # Exit the process completely
    logger.info("Exiting...")


async def wait_for_quit(http3_task, http1_task):
    """Wait for user to enter 'q' and then shut down servers."""
    loop = asyncio.get_running_loop()

    def _input():
        return sys.stdin.readline()

    print("Press 'q' + Enter to quit.")
    while True:
        # Run blocking input in executor
        user_input = await loop.run_in_executor(None, _input)
        if user_input.strip().lower() == "q":
            await shutdown(http3_task, http1_task)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTTP/1.1 and HTTP/3 server")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to listen on"
    )
    parser.add_argument(
        "--http1-port", type=int, default=8443, help="Port for HTTP/1.1"
    )
    parser.add_argument("--http3-port", type=int, default=4433, help="Port for HTTP/3")
    parser.add_argument(
        "--cert", type=str, default="./certs/server.crt", help="TLS certificate file"
    )
    parser.add_argument(
        "--key", type=str, default="./certs/server.key", help="TLS key file"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging based on verbosity flag
    configure_logger(args.verbose)

    # Check if certificates exist
    if not os.path.exists(args.cert) or not os.path.exists(args.key):
        logger.error(f"Certificate files not found. Please run makecert.sh first.")
        sys.exit(1)

    # Check if ports are in use
    import socket

    def is_port_in_use(host, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return False
            except OSError:
                return True

    if is_port_in_use(args.host, args.http3_port):
        logger.error(
            f"HTTP/3 port {args.http3_port} is already in use. Please kill any processes using this port."
        )
        sys.exit(1)

    if is_port_in_use(args.host, args.http1_port):
        logger.error(
            f"HTTP/1.1 port {args.http1_port} is already in use. Please kill any processes using this port."
        )
        sys.exit(1)

    try:
        asyncio.run(run_both_servers(args))
        print("Exited gracefully.")
    except KeyboardInterrupt:
        logger.info("Server stopped")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error running server: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
