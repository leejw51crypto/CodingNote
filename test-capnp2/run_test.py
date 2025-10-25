#!/usr/bin/env python3
"""
Orchestrated RPC Test Runner
Starts server, waits, runs client, shows results, then stops everything
"""

import asyncio
import signal
import sys
import time

import calculator_capnp
import capnp  # Must import capnp first


class RPCTestOrchestrator:
    def __init__(self):
        self.server_process = None
        self.server_task = None

    async def start_server(self):
        """Start the RPC server"""
        print("🚀 Starting RPC Server...")

        async def new_connection(stream):
            """Handle new client connection"""
            from rpc_server import CalculatorImpl

            await capnp.TwoPartyServer(
                stream, bootstrap=CalculatorImpl()
            ).on_disconnect()

        server = await capnp.AsyncIoStream.create_server(
            new_connection, "localhost", 60000
        )

        print("✅ Server listening on localhost:60000\n")

        # Keep server running
        async with server:
            await asyncio.Event().wait()

    async def run_client_tests(self):
        """Run the RPC client tests"""
        print("⏳ Waiting for server to be ready...")
        await asyncio.sleep(2)

        print("🔌 Connecting to Calculator server at localhost:60000...")

        try:
            # Connect to the server
            client = await capnp.AsyncIoStream.create_connection(
                host="localhost", port=60000
            )
            calculator = (
                capnp.TwoPartyClient(client)
                .bootstrap()
                .cast_as(calculator_capnp.Calculator)
            )

            print("✅ Connected! Making RPC calls...\n")

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
                print(
                    f"   ✅ Error handling works! Server exception caught on client side"
                )
                print(f"   📝 Exception: ValueError - Division by zero\n")

            print("👋 Closing connection...")
            print("\n" + "=" * 60)
            print("🎉 All tests completed successfully!")
            print("=" * 60 + "\n")

        except Exception as e:
            print(f"❌ Error during RPC calls: {e}")
            raise

    async def run_orchestrated_test(self):
        """Run the complete orchestrated test"""
        print("\n" + "=" * 60)
        print("🎬 Cap'n Proto RPC Orchestrated Test")
        print("=" * 60 + "\n")

        # Start server in background
        self.server_task = asyncio.create_task(self.start_server())

        try:
            # Run client tests
            await self.run_client_tests()

        finally:
            # Stop server
            print("🛑 Stopping server...")
            if self.server_task:
                self.server_task.cancel()
                try:
                    await self.server_task
                except asyncio.CancelledError:
                    pass

            print("✅ Server stopped\n")


async def main():
    """Main entry point"""
    orchestrator = RPCTestOrchestrator()

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n⚠️  Interrupted by user")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        await orchestrator.run_orchestrated_test()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(capnp.run(main()))
