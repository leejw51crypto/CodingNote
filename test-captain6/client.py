#!/usr/bin/env python3

import argparse
import asyncio
import capnp
import hello_capnp


async def run_client(host, port):
    # Connect to the server
    client = capnp.TwoPartyClient(
        await capnp.AsyncIoStream.create_connection(host=host, port=port)
    )

    # Get the Hello interface using bootstrap
    hello = client.bootstrap().cast_as(hello_capnp.Hello)

    # Call getCurrentTime method
    print("\n== 🕒 Get Current Time 🕒 ==")
    time_request = hello.getCurrentTime_request()
    time_result = await time_request.send()
    print(f"🌍 Local time: {time_result.localTime}")
    print(f"🌐 UTC time: {time_result.utcTime}")

    # Call greeting method with a name
    print("\n== 👋 Greeting 👋 ==")
    name = "Cap'n Proto User"
    greeting_request = hello.greeting_request()
    greeting_request.name = name
    greeting_result = await greeting_request.send()
    print(f"💬 Server says: {greeting_result.message}")

    # Call helloWorld method
    print("\n== 🌟 Hello World Data 🌟 ==")
    count = 5
    hello_request = hello.helloWorld_request()
    hello_request.count = count
    hello_result = await hello_request.send()
    data = hello_result.result

    print(f"📝 Message: {data.message}")
    print(f"⏱️ Timestamp: {data.timestamp}")
    print(f"🔢 Double value: {data.doubleValue}")
    print(f"📊 Int array: {data.intArray}")
    print(f"🔐 Binary data (first 10 bytes): {data.binaryData[:10]}")
    print("\n✨ All requests completed successfully! ✨")


def parse_args():
    parser = argparse.ArgumentParser(
        usage="Connect to Hello server at the given address/port"
    )
    parser.add_argument("host", help="HOST:PORT")
    return parser.parse_args()


async def cmd_main(host_port):
    host, port = host_port.split(":")
    print(f"🚀 Connecting to Hello server at {host}:{port}...")
    await run_client(host, port)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(capnp.run(cmd_main(args.host)))
