#!/usr/bin/env python3
"""
Cap'n Proto RPC Server Example
Implements a Calculator service that performs basic arithmetic operations.
"""

import asyncio

import capnp  # Must import capnp first
import calculator_capnp


class CalculatorImpl(calculator_capnp.Calculator.Server):
    """Implementation of the Calculator RPC interface"""

    async def add(self, a, b, **kwargs):
        """Add two numbers"""
        result = a + b
        print(f"Server: add({a}, {b}) = {result}")
        return result

    async def subtract(self, a, b, **kwargs):
        """Subtract two numbers"""
        result = a - b
        print(f"Server: subtract({a}, {b}) = {result}")
        return result

    async def multiply(self, a, b, **kwargs):
        """Multiply two numbers"""
        result = a * b
        print(f"Server: multiply({a}, {b}) = {result}")
        return result

    async def divide(self, a, b, **kwargs):
        """Divide two numbers"""
        if b == 0:
            raise ValueError("Division by zero")
        result = float(a) / float(b)
        print(f"Server: divide({a}, {b}) = {result}")
        return result

    async def evaluate(self, expression, **kwargs):
        """Evaluate a mathematical expression (simple implementation)"""
        try:
            # WARNING: eval() is unsafe for production - use a proper parser
            result = float(eval(expression, {"__builtins__": {}}, {}))
            print(f"Server: evaluate('{expression}') = {result}")
            return result
        except Exception as e:
            print(f"Server: evaluate('{expression}') failed: {e}")
            raise ValueError(f"Invalid expression: {e}")


async def new_connection(stream):
    """Handle new client connection"""
    await capnp.TwoPartyServer(stream, bootstrap=CalculatorImpl()).on_disconnect()


async def main():
    """Start the RPC server"""
    host = "localhost"
    port = 60000

    print(f"Calculator RPC server listening on {host}:{port}")
    print("Press Ctrl+C to stop the server")

    # Start the server
    server = await capnp.AsyncIoStream.create_server(new_connection, host, port)

    async with server:
        # Keep the server running indefinitely
        await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        # Create and run the event loop
        asyncio.run(capnp.run(main()))
    except KeyboardInterrupt:
        print("\nServer stopped")
