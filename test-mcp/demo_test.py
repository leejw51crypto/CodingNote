#!/usr/bin/env python3
"""
Demo script for MyNote MCP Server
Shows the server functionality in action
"""

import asyncio
import sys

sys.path.append(".")

from mynote_server import handle_call_tool, handle_list_tools


async def demo_mynote_server():
    """Demonstrate the MyNote MCP server functionality"""

    print("🚀 MyNote MCP Server Demo")
    print("=" * 40)

    # Test 1: Write some data
    print("\n📝 Testing WRITE functionality:")

    test_data = [
        ("greeting", "Hello, World!"),
        ("note1", "This is my first note"),
        ("config", '{"theme": "dark", "language": "en"}'),
        ("todo", "Buy groceries\nCall mom\nFinish project"),
    ]

    for key, value in test_data:
        result = await handle_call_tool("write", {"key": key, "value": value})
        print(f"  ✅ {result[0].text}")

    # Test 2: Read the data back
    print("\n📖 Testing READ functionality:")

    for key, _ in test_data:
        result = await handle_call_tool("read", {"key": key})
        print(f"  📄 Reading '{key}':")
        print(f"     {result[0].text.replace(chr(10), chr(10) + '     ')}")
        print()

    # Test 3: Test error cases
    print("🚨 Testing ERROR handling:")

    # Try to read non-existent key
    result = await handle_call_tool("read", {"key": "non_existent"})
    print(f"  ❌ {result[0].text}")

    # Try to write without key
    result = await handle_call_tool("write", {"value": "orphan value"})
    print(f"  ❌ {result[0].text}")

    # Test 4: Get current time
    print("\n🕐 Testing GETTIME functionality:")
    result = await handle_call_tool("gettime", {})
    print(f"  🕐 {result[0].text}")

    # Test 5: List available tools
    print("\n🛠️  Available tools:")
    tools = await handle_list_tools()
    for tool in tools:
        print(f"  • {tool.name}: {tool.description}")

    # Test 6: Overwrite existing data
    print("\n🔄 Testing data OVERWRITE:")
    original_result = await handle_call_tool("read", {"key": "greeting"})
    print(
        f"  📖 Original: {original_result[0].text.split('Value: ')[1].split(chr(10))[0]}"
    )

    await handle_call_tool(
        "write", {"key": "greeting", "value": "Hello, Updated World!"}
    )
    updated_result = await handle_call_tool("read", {"key": "greeting"})
    print(
        f"  📖 Updated: {updated_result[0].text.split('Value: ')[1].split(chr(10))[0]}"
    )

    print("\n✨ Demo completed successfully!")


if __name__ == "__main__":
    print("Starting MyNote MCP Server Demo...")
    asyncio.run(demo_mynote_server())
