#!/usr/bin/env python3

import argparse
import asyncio
import datetime
import time
import random
import capnp
import hello_capnp


class HelloImpl(hello_capnp.Hello.Server):
    """Implementation of the Hello Cap'n Proto interface."""

    async def getCurrentTime(self, _context, **kwargs):
        """Get current time in both local and UTC."""
        now = datetime.datetime.now()
        utc_now = datetime.datetime.now(datetime.UTC)

        print(f"🕒 Sending time information to client...")
        # Use _context.results to set the return values
        _context.results.localTime = now.strftime("%Y-%m-%d %H:%M:%S")
        _context.results.utcTime = utc_now.strftime("%Y-%m-%d %H:%M:%S")

    async def greeting(self, name, _context, **kwargs):
        """Simple greeting with a name."""
        print(f"👋 Greeting request from: {name}")
        _context.results.message = f"Hello, {name}! ✨"

    async def helloWorld(self, count, _context, **kwargs):
        """Return HelloWorldData with message, timestamp, and other data."""
        print(f"🌟 Processing helloWorld request with count: {count}")
        # Generate some sample data
        random_doubles = random.random() * 100
        int_array = [i * 10 for i in range(count)]
        binary_data = bytes([random.randint(0, 255) for _ in range(32)])

        # Build the HelloWorldData structure
        result = _context.results.result
        result.message = "✨ Hello, World! ✨"
        result.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        result.doubleValue = random_doubles
        result.intArray = int_array
        result.binaryData = binary_data
        print("🚀 Sending HelloWorldData response...")


async def new_connection(stream):
    print("🔌 New client connected!")
    server = capnp.TwoPartyServer(stream, bootstrap=HelloImpl())
    await server.on_disconnect()
    print("🔌 Client disconnected")


def parse_args():
    parser = argparse.ArgumentParser(
        usage="Runs the Hello server at the given address/port"
    )
    parser.add_argument("address", help="ADDRESS:PORT")
    return parser.parse_args()


async def main():
    args = parse_args()
    host, port = args.address.split(":")
    server = await capnp.AsyncIoStream.create_server(new_connection, host, port)
    print(f"🚀 Server running at {host}:{port}")
    print("⭐ Ready to accept connections...")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(capnp.run(main()))
