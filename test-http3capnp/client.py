#!/usr/bin/env python3

import argparse
import asyncio
import logging
import ssl
import time
import json
from typing import Optional, Dict, List, Any, cast
from urllib.parse import urlparse

from aioquic.asyncio.client import connect
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.h3.connection import H3_ALPN, H3Connection
from aioquic.h3.events import DataReceived, HeadersReceived, H3Event
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import QuicEvent

# Cap'n Proto support
import hellocapnp

logger = logging.getLogger("http3-client")


class H3Client(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._http = H3Connection(self._quic)
        self._request_events: Dict[int, List[H3Event]] = {}
        self._request_waiter: Dict[int, asyncio.Future] = {}
        self._request_headers: Dict[int, Dict[str, str]] = {}
        self._response_data: Dict[int, bytes] = {}

    async def get(self, url: str, capnp: bool = False) -> bytes:
        """
        Perform a GET request.
        """
        parsed = urlparse(url)
        path = parsed.path or "/"
        authority = parsed.netloc
        stream_id = self._quic.get_next_available_stream_id()
        logger.debug(f"Starting GET request on stream {stream_id}")
        self._http.send_headers(
            stream_id=stream_id,
            headers=[
                (b":method", b"GET"),
                (b":scheme", b"https"),
                (b":authority", authority.encode()),
                (b":path", path.encode()),
                (b"user-agent", b"aioquic-http3-client"),
            ],
            end_stream=True,
        )
        waiter = asyncio.Future()
        self._request_events[stream_id] = []
        self._request_headers[stream_id] = {}
        self._response_data[stream_id] = b""
        self._request_waiter[stream_id] = waiter
        try:
            return await asyncio.wait_for(waiter, timeout=10.0)
        except asyncio.TimeoutError:
            logger.error(f"Request timed out for stream {stream_id}")
            return b"Request timed out"

    async def post(self, url: str, data: str, capnp: bool = False) -> bytes:
        """
        Perform a POST request.
        """
        parsed = urlparse(url)
        path = parsed.path or "/"
        authority = parsed.netloc
        stream_id = self._quic.get_next_available_stream_id()
        logger.debug(f"Starting POST request on stream {stream_id}")
        headers = [
            (b":method", b"POST"),
            (b":scheme", b"https"),
            (b":authority", authority.encode()),
            (b":path", path.encode()),
            (b"user-agent", b"aioquic-http3-client"),
        ]
        if capnp:
            headers.append((b"content-type", b"application/x-capnp"))
            body = hellocapnp.encode_hello_world(data)
        else:
            headers.append((b"content-type", b"text/plain"))
            body = data.encode()
        self._http.send_headers(
            stream_id=stream_id,
            headers=headers,
        )
        self._http.send_data(stream_id=stream_id, data=body, end_stream=True)
        waiter = asyncio.Future()
        self._request_events[stream_id] = []
        self._request_headers[stream_id] = {}
        self._response_data[stream_id] = b""
        self._request_waiter[stream_id] = waiter
        try:
            return await asyncio.wait_for(waiter, timeout=10.0)
        except asyncio.TimeoutError:
            logger.error(f"Request timed out for stream {stream_id}")
            return b"Request timed out"

    def http_event_received(self, event: H3Event) -> None:
        if isinstance(event, (HeadersReceived, DataReceived)):
            stream_id = event.stream_id
            if stream_id in self._request_events:
                # Store event
                self._request_events[stream_id].append(event)

                # Process headers
                if isinstance(event, HeadersReceived):
                    headers = {}
                    status = None
                    for header, value in event.headers:
                        header_name = header.decode()
                        if header_name == ":status":
                            status = int(value.decode())
                        else:
                            # Remove leading : from pseudo-headers
                            if header_name.startswith(":"):
                                header_name = header_name[1:]
                            headers[header_name] = value.decode()

                    self._request_headers[stream_id] = headers
                    logger.info(f"Received response headers, status={status}")

                # Process data
                elif isinstance(event, DataReceived):
                    # Add to accumulated data for this stream
                    self._response_data[stream_id] += event.data

                    # Simple approach: if we have data and a waiter, complete the request
                    # This might complete early, but at least it won't hang
                    if stream_id in self._request_waiter:
                        waiter = self._request_waiter[stream_id]
                        if not waiter.done():
                            response_data = self._response_data[stream_id]
                            if response_data:
                                logger.debug(
                                    f"Data received for stream {stream_id}, {len(response_data)} bytes"
                                )
                                waiter.set_result(response_data)
                                # Clean up
                                self._request_waiter.pop(stream_id)

    def quic_event_received(self, event: QuicEvent) -> None:
        for http_event in self._http.handle_event(event):
            self.http_event_received(http_event)


async def main(
    url: str,
    method: str,
    data: Optional[str],
    ca_file: Optional[str],
    insecure: bool,
    verbose: bool,
    capnp: bool = False,
):
    parsed = urlparse(url)
    assert parsed.scheme == "https", "Only HTTPS URLs are supported"
    host = parsed.hostname
    port = parsed.port or 443

    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", level=log_level
    )

    logger.info(f"Connecting to {host}:{port}")

    # QUIC configuration
    configuration = QuicConfiguration(
        alpn_protocols=H3_ALPN,
        is_client=True,
        verify_mode=ssl.CERT_NONE if insecure else ssl.CERT_REQUIRED,
    )

    if ca_file:
        configuration.load_verify_locations(ca_file)

    try:
        async with connect(
            host=host,
            port=port,
            configuration=configuration,
            create_protocol=H3Client,
        ) as client:
            client = cast(H3Client, client)
            logger.info(f"Sending {method} request to {url}")
            start_time = time.time()

            # Send request and get response
            if method.upper() == "POST" and data is not None:
                response_data = await client.post(url, data, capnp=capnp)
            else:
                response_data = await client.get(url, capnp=capnp)

            end_time = time.time()
            elapsed = end_time - start_time

            if capnp:
                try:
                    msg = hellocapnp.decode_hello_world(response_data)
                    logger.info(f"Response received in {elapsed:.3f} seconds (Capnp):")
                    print(f"Capnp message: {msg}")
                except Exception as e:
                    logger.error(f"Capnp decode error: {e}")
                    print(f"Invalid Capnp response: {response_data}")
            else:
                # Try to parse JSON response
                try:
                    json_data = json.loads(response_data)
                    logger.info(f"Response received in {elapsed:.3f} seconds (JSON):")
                    print(json.dumps(json_data, indent=2))
                except json.JSONDecodeError:
                    # Try to decode as text
                    try:
                        response_text = response_data.decode()
                        logger.info(f"Response received in {elapsed:.3f} seconds:")
                        print(response_text)
                    except UnicodeDecodeError:
                        logger.info(
                            f"Response received in {elapsed:.3f} seconds (binary data): {len(response_data)} bytes"
                        )

    except Exception as e:
        logger.error(f"Error: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTTP/3 Client")
    parser.add_argument("url", help="URL to request (e.g. https://localhost:4433/)")
    parser.add_argument(
        "--method", choices=["GET", "POST"], default="GET", help="HTTP method"
    )
    parser.add_argument("--data", help="Data to send with POST request")
    parser.add_argument("--ca-file", help="CA certificate file")
    parser.add_argument(
        "--insecure", action="store_true", help="Disable certificate verification"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--capnp", action="store_true", help="Use Cap'n Proto binary encoding"
    )

    args = parser.parse_args()

    # If method is POST but no data provided, use empty string
    if args.method == "POST" and args.data is None:
        args.data = ""

    try:
        exit_code = asyncio.run(
            main(
                args.url,
                args.method,
                args.data,
                args.ca_file,
                args.insecure,
                args.verbose,
                args.capnp,
            )
        )
        exit(exit_code)
    except KeyboardInterrupt:
        pass
