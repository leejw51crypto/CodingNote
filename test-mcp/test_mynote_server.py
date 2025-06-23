#!/usr/bin/env python3
"""
Test suite for MyNote MCP Server
Tests all functionality including write, read, gettime, and error handling
"""

import asyncio
import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

# Import the server functions
import sys

sys.path.append(".")
from mynote_server import (
    server,
    get_db_file_path,
    DB_DIR,
    handle_call_tool,
    handle_list_tools,
)


class TestMyNoteServer(unittest.TestCase):
    """Test class for MyNote MCP Server"""

    def setUp(self):
        """Set up test environment before each test"""
        # Create a temporary directory for testing
        self.test_db_dir = Path(tempfile.mkdtemp())

        # Patch the DB_DIR to use our test directory
        self.db_dir_patcher = patch("mynote_server.DB_DIR", self.test_db_dir)
        self.db_dir_patcher.start()

    def tearDown(self):
        """Clean up after each test"""
        # Remove test directory
        if self.test_db_dir.exists():
            shutil.rmtree(self.test_db_dir)

        # Stop patching
        self.db_dir_patcher.stop()

    def test_get_db_file_path(self):
        """Test the database file path generation"""
        # Test normal key
        path = get_db_file_path("test_key")
        expected = self.test_db_dir / "test_key.json"
        self.assertEqual(path, expected)

        # Test key with special characters (should be sanitized)
        path = get_db_file_path("test/key@#$")
        expected = self.test_db_dir / "testkey.json"
        self.assertEqual(path, expected)

        # Test key with allowed special characters
        path = get_db_file_path("test_key-1.backup")
        expected = self.test_db_dir / "test_key-1.backup.json"
        self.assertEqual(path, expected)

    async def test_write_tool(self):
        """Test the write tool functionality"""
        # Test successful write
        result = await handle_call_tool(
            "write", {"key": "test_key", "value": "test_value"}
        )

        self.assertEqual(len(result), 1)
        self.assertIn("Successfully stored key 'test_key'", result[0].text)

        # Verify file was created
        file_path = get_db_file_path("test_key")
        self.assertTrue(file_path.exists())

        # Verify file contents
        with open(file_path, "r") as f:
            data = json.load(f)

        self.assertEqual(data["key"], "test_key")
        self.assertEqual(data["value"], "test_value")
        self.assertIn("timestamp", data)

    async def test_write_tool_errors(self):
        """Test write tool error handling"""
        # Test missing arguments
        result = await handle_call_tool("write", None)
        self.assertIn("Error: Missing arguments", result[0].text)

        # Test missing key
        result = await handle_call_tool("write", {"value": "test_value"})
        self.assertIn("Error: Both key and value are required", result[0].text)

        # Test missing value
        result = await handle_call_tool("write", {"key": "test_key"})
        self.assertIn("Error: Both key and value are required", result[0].text)

        # Test empty key
        result = await handle_call_tool("write", {"key": "", "value": "test_value"})
        self.assertIn("Error: Both key and value are required", result[0].text)

    async def test_read_tool(self):
        """Test the read tool functionality"""
        # First write some data
        await handle_call_tool("write", {"key": "read_test", "value": "read_value"})

        # Test successful read
        result = await handle_call_tool("read", {"key": "read_test"})

        self.assertEqual(len(result), 1)
        self.assertIn("Key: read_test", result[0].text)
        self.assertIn("Value: read_value", result[0].text)
        self.assertIn("Stored at:", result[0].text)

    async def test_read_tool_errors(self):
        """Test read tool error handling"""
        # Test missing arguments
        result = await handle_call_tool("read", None)
        self.assertIn("Error: Missing arguments", result[0].text)

        # Test missing key
        result = await handle_call_tool("read", {})
        self.assertIn("Error: Key is required", result[0].text)

        # Test non-existent key
        result = await handle_call_tool("read", {"key": "non_existent_key"})
        self.assertIn("Key 'non_existent_key' not found in database", result[0].text)

    async def test_gettime_tool(self):
        """Test the gettime tool functionality"""
        result = await handle_call_tool("gettime", {})

        self.assertEqual(len(result), 1)
        self.assertIn("Local time:", result[0].text)
        self.assertIn("UTC time:", result[0].text)

        # Test with arguments (should still work)
        result = await handle_call_tool("gettime", {"ignored": "parameter"})
        self.assertEqual(len(result), 1)
        self.assertIn("Local time:", result[0].text)

    async def test_unknown_tool(self):
        """Test unknown tool handling"""
        result = await handle_call_tool("unknown_tool", {})
        self.assertIn("Unknown tool: unknown_tool", result[0].text)

    async def test_list_tools(self):
        """Test the list_tools functionality"""
        tools = await handle_list_tools()

        self.assertEqual(len(tools), 3)

        tool_names = [tool.name for tool in tools]
        self.assertIn("write", tool_names)
        self.assertIn("read", tool_names)
        self.assertIn("gettime", tool_names)

        # Check write tool schema
        write_tool = next(tool for tool in tools if tool.name == "write")
        self.assertEqual(
            write_tool.description, "Store a key-value pair in the database"
        )
        self.assertIn("key", write_tool.inputSchema["properties"])
        self.assertIn("value", write_tool.inputSchema["properties"])

        # Check read tool schema
        read_tool = next(tool for tool in tools if tool.name == "read")
        self.assertEqual(
            read_tool.description, "Retrieve a value by its key from the database"
        )
        self.assertIn("key", read_tool.inputSchema["properties"])

        # Check gettime tool schema
        gettime_tool = next(tool for tool in tools if tool.name == "gettime")
        self.assertEqual(gettime_tool.description, "Get current local and UTC time")

    async def test_write_read_cycle(self):
        """Test complete write-read cycle with various data types"""
        test_cases = [
            ("simple", "Hello World"),
            ("json_like", '{"nested": "value"}'),
            ("multiline", "Line 1\nLine 2\nLine 3"),
            ("special_chars", "Special: !@#$%^&*()"),
            ("unicode", "Unicode: 🚀 ñ α β γ"),
            ("numbers", "42"),
        ]

        for key, value in test_cases:
            with self.subTest(key=key, value=value):
                # Write
                write_result = await handle_call_tool(
                    "write", {"key": key, "value": value}
                )
                self.assertIn("Successfully stored", write_result[0].text)

                # Read
                read_result = await handle_call_tool("read", {"key": key})
                self.assertIn(f"Key: {key}", read_result[0].text)
                self.assertIn(f"Value: {value}", read_result[0].text)

        # Test empty value separately
        write_result = await handle_call_tool(
            "write", {"key": "empty_test", "value": ""}
        )
        # Empty string should be treated as valid but trigger the error
        self.assertIn("Error: Both key and value are required", write_result[0].text)

    async def test_overwrite_existing_key(self):
        """Test overwriting an existing key"""
        key = "overwrite_test"

        # Write initial value
        await handle_call_tool("write", {"key": key, "value": "initial_value"})

        # Read initial value
        result = await handle_call_tool("read", {"key": key})
        self.assertIn("Value: initial_value", result[0].text)

        # Overwrite with new value
        await handle_call_tool("write", {"key": key, "value": "new_value"})

        # Read new value
        result = await handle_call_tool("read", {"key": key})
        self.assertIn("Value: new_value", result[0].text)
        self.assertNotIn("initial_value", result[0].text)


async def run_async_tests():
    """Run all async tests"""
    suite = unittest.TestSuite()

    # Create test instance
    test_instance = TestMyNoteServer()

    # List of async test methods
    async_tests = [
        "test_write_tool",
        "test_write_tool_errors",
        "test_read_tool",
        "test_read_tool_errors",
        "test_gettime_tool",
        "test_unknown_tool",
        "test_list_tools",
        "test_write_read_cycle",
        "test_overwrite_existing_key",
    ]

    print("Running MyNote MCP Server Tests")
    print("=" * 50)

    passed = 0
    failed = 0

    for test_name in async_tests:
        test_instance.setUp()
        try:
            print(f"Running {test_name}...", end=" ")
            await getattr(test_instance, test_name)()
            print("✅ PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed += 1
        finally:
            test_instance.tearDown()

    # Run sync tests
    sync_tests = ["test_get_db_file_path"]
    for test_name in sync_tests:
        test_instance.setUp()
        try:
            print(f"Running {test_name}...", end=" ")
            getattr(test_instance, test_name)()
            print("✅ PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed += 1
        finally:
            test_instance.tearDown()

    print("=" * 50)
    print(f"Tests completed: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {failed} test(s) failed")


if __name__ == "__main__":
    asyncio.run(run_async_tests())
