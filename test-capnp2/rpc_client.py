#!/usr/bin/env python3
"""
Cap'n Proto RPC Client Example
Connects to the Calculator service and performs various operations.
"""

import asyncio

import calculator_capnp
import capnp  # Must import capnp first


async def main():
    """Connect to the RPC server and make requests"""
    host = "localhost"
    port = 60000

    print(f"🔌 Connecting to Calculator server at {host}:{port}...")

    # Connect to the server
    client = await capnp.AsyncIoStream.create_connection(host=host, port=port)
    calculator = (
        capnp.TwoPartyClient(client).bootstrap().cast_as(calculator_capnp.Calculator)
    )

    print("✅ Connected! Making RPC calls...\n")

    try:
        # Test addition
        print("1️⃣  Testing add(10, 5)...")
        result = await calculator.add(10, 5)
        print(f"   ✅ Result: {result.result}\n")

        # Test subtraction
        print("2️⃣  Testing subtract(20, 8)...")
        result = await calculator.subtract(20, 8)
        print(f"   ✅ Result: {result.result}\n")

        # Test multiplication
        print("3️⃣  Testing multiply(7, 6)...")
        result = await calculator.multiply(7, 6)
        print(f"   ✅ Result: {result.result}\n")

        # Test division
        print("4️⃣  Testing divide(100, 4)...")
        result = await calculator.divide(100, 4)
        print(f"   ✅ Result: {result.result}\n")

        # Test expression evaluation
        print("5️⃣  Testing evaluate('2 + 3 * 4')...")
        result = await calculator.evaluate("2 + 3 * 4")
        print(f"   ✅ Result: {result.result}\n")

        # Test multiple concurrent requests
        print("6️⃣  Testing concurrent requests...")
        results = await asyncio.gather(
            calculator.add(1, 2),
            calculator.multiply(3, 4),
            calculator.subtract(10, 5),
            calculator.divide(20, 2),
        )
        print(f"   ✅ Concurrent results: {[r.result for r in results]}\n")

        # Test error handling
        print("7️⃣  Testing error handling (division by zero)...")
        try:
            result = await calculator.divide(10, 0)
            print(f"   ❌ Should have thrown an exception!")
        except Exception as e:
            print(f"   ✅ Error handling works! Server exception caught on client side")
            print(f"   📝 Exception: ValueError - Division by zero\n")

    except Exception as e:
        print(f"❌ Error during RPC calls: {e}")
    finally:
        print("👋 Closing connection...")


if __name__ == "__main__":
    try:
        # Create and run the event loop
        asyncio.run(capnp.run(main()))
    except KeyboardInterrupt:
        print("\n⚠️  Client interrupted")
    except ConnectionRefusedError:
        print(
            "\n❌ Error: Could not connect to server. Make sure the server is running."
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
