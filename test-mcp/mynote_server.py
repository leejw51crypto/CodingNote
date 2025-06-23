#!/usr/bin/env python3
"""
MyNote MCP Server - A simple key-value database with file storage
Provides tools, resources, and prompts functionality
"""

import json
import os
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import mcp.types as types
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio


# Database directory for storing key-value pairs
# Use a writable location in the user's home directory
# Can be overridden with MYNOTE_DB_PATH environment variable
DB_DIR = Path(os.getenv("MYNOTE_DB_PATH", Path.home() / ".mynote_db"))
DB_DIR.mkdir(exist_ok=True)

# Initialize the MCP server
server = Server("mynote")


def get_db_file_path(key: str) -> Path:
    """Get the file path for a given key"""
    # Sanitize the key to be filesystem-safe
    safe_key = "".join(c for c in key if c.isalnum() or c in "._-")
    return DB_DIR / f"{safe_key}.json"


def get_all_keys() -> list[str]:
    """Get all stored keys from the database"""
    keys = []
    for file_path in DB_DIR.glob("*.json"):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                keys.append(data["key"])
        except Exception:
            continue
    return sorted(keys)


def get_db_stats() -> dict[str, Any]:
    """Get database statistics"""
    keys = get_all_keys()
    total_size = sum(f.stat().st_size for f in DB_DIR.glob("*.json"))

    return {
        "total_keys": len(keys),
        "total_size_bytes": total_size,
        "database_path": str(DB_DIR),
        "keys": keys,
    }


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="write",
            description="Store a key-value pair in the database",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key to store the value under",
                    },
                    "value": {"type": "string", "description": "The value to store"},
                },
                "required": ["key", "value"],
            },
        ),
        types.Tool(
            name="read",
            description="Retrieve a value by its key from the database",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key to retrieve the value for",
                    }
                },
                "required": ["key"],
            },
        ),
        types.Tool(
            name="gettime",
            description="Get current local and UTC time",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ),
    ]


@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """List available resources"""
    resources = [
        types.Resource(
            uri="mynote://stats",
            name="Database Statistics",
            description="Statistics about the MyNote database",
            mimeType="application/json",
        ),
        types.Resource(
            uri="mynote://keys",
            name="All Keys",
            description="List of all stored keys in the database",
            mimeType="application/json",
        ),
    ]

    # Add individual resources for each stored key
    for key in get_all_keys():
        resources.append(
            types.Resource(
                uri=f"mynote://key/{key}",
                name=f"Note: {key}",
                description=f"Content of note with key '{key}'",
                mimeType="application/json",
            )
        )

    return resources


@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read resource content"""

    if uri == "mynote://stats":
        stats = get_db_stats()
        return json.dumps(stats, indent=2)

    elif uri == "mynote://keys":
        keys = get_all_keys()
        return json.dumps({"keys": keys}, indent=2)

    elif uri.startswith("mynote://key/"):
        key = uri.replace("mynote://key/", "")
        file_path = get_db_file_path(key)

        if not file_path.exists():
            raise Exception(f"Key '{key}' not found in database")

        with open(file_path, "r") as f:
            data = json.load(f)

        return json.dumps(data, indent=2)

    else:
        raise Exception(f"Unknown resource URI: {uri}")


@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    """List available prompts"""
    return [
        types.Prompt(
            name="create_note",
            description="Help create a new note with proper formatting",
            arguments=[
                types.PromptArgument(
                    name="topic",
                    description="The main topic or subject of the note",
                    required=True,
                ),
                types.PromptArgument(
                    name="content_type",
                    description="Type of content (e.g., 'meeting', 'idea', 'todo', 'reference')",
                    required=False,
                ),
            ],
        ),
        types.Prompt(
            name="query_notes",
            description="Help search and query existing notes",
            arguments=[
                types.PromptArgument(
                    name="search_term",
                    description="What to search for in the notes",
                    required=True,
                ),
                types.PromptArgument(
                    name="context",
                    description="Additional context for the search",
                    required=False,
                ),
            ],
        ),
        types.Prompt(
            name="organize_notes",
            description="Help organize and categorize existing notes",
            arguments=[
                types.PromptArgument(
                    name="organization_method",
                    description="How to organize (e.g., 'by_date', 'by_topic', 'by_priority')",
                    required=False,
                ),
            ],
        ),
    ]


@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """Get prompt content"""

    if name == "create_note":
        topic = arguments.get("topic", "") if arguments else ""
        content_type = (
            arguments.get("content_type", "general") if arguments else "general"
        )

        prompt_content = f"""You are helping to create a new note in the MyNote database.

Topic: {topic}
Content Type: {content_type}

Please help structure this note with:
1. A clear, descriptive key that follows naming conventions
2. Well-organized content that includes:
   - Main points or ideas
   - Relevant details
   - Any action items or next steps
   - Timestamps or dates if relevant

Consider the content type '{content_type}' when structuring the note.
Make sure the key is filesystem-safe (alphanumeric, dots, dashes, underscores only).

Use the 'write' tool to store the note when ready."""

        return types.GetPromptResult(
            description=f"Create a {content_type} note about {topic}",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=prompt_content),
                ),
            ],
        )

    elif name == "query_notes":
        search_term = arguments.get("search_term", "") if arguments else ""
        context = arguments.get("context", "") if arguments else ""

        # Get current notes for context
        keys = get_all_keys()
        keys_list = ", ".join(keys[:10])  # Show first 10 keys
        total_keys = len(keys)

        prompt_content = f"""You are helping to search and query notes in the MyNote database.

Search Term: {search_term}
Context: {context}

Current database contains {total_keys} notes with keys like: {keys_list}

To help find relevant information:
1. Use the 'read' tool to examine specific notes
2. Look for notes with keys that might relate to "{search_term}"
3. Consider variations and related terms
4. Provide a summary of findings

Start by reading the most promising note keys, then provide insights about what was found."""

        return types.GetPromptResult(
            description=f"Search for notes related to: {search_term}",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=prompt_content),
                ),
            ],
        )

    elif name == "organize_notes":
        organization_method = (
            arguments.get("organization_method", "by_topic")
            if arguments
            else "by_topic"
        )

        # Get current notes for context
        keys = get_all_keys()
        stats = get_db_stats()

        prompt_content = f"""You are helping to organize notes in the MyNote database.

Organization Method: {organization_method}
Current Status:
- Total Notes: {stats['total_keys']}
- Database Size: {stats['total_size_bytes']} bytes
- Sample Keys: {', '.join(keys[:15])}

Please help organize these notes by:
1. Analyzing the existing note structure and content
2. Suggesting improvements based on the '{organization_method}' approach
3. Identifying any duplicate or redundant notes
4. Recommending a consistent naming convention
5. Grouping related notes together

Use the resource 'mynote://stats' and individual note resources to analyze the content.
Consider creating summary notes or index notes for better organization."""

        return types.GetPromptResult(
            description=f"Organize notes using method: {organization_method}",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=prompt_content),
                ),
            ],
        )

    else:
        raise Exception(f"Unknown prompt: {name}")


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any] | None
) -> list[types.TextContent]:
    """Handle tool calls"""

    if name == "write":
        if arguments is None:
            return [types.TextContent(type="text", text="Error: Missing arguments")]

        key = arguments.get("key")
        value = arguments.get("value")

        if not key or not value:
            return [
                types.TextContent(
                    type="text", text="Error: Both key and value are required"
                )
            ]

        try:
            # Store the key-value pair in a JSON file
            file_path = get_db_file_path(key)
            data = {"key": key, "value": value, "timestamp": datetime.now().isoformat()}

            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)

            return [
                types.TextContent(
                    type="text",
                    text=f"Successfully stored key '{key}' with value '{value}'",
                )
            ]

        except Exception as e:
            return [
                types.TextContent(
                    type="text", text=f"Error storing key-value pair: {str(e)}"
                )
            ]

    elif name == "read":
        if arguments is None:
            return [types.TextContent(type="text", text="Error: Missing arguments")]

        key = arguments.get("key")

        if not key:
            return [types.TextContent(type="text", text="Error: Key is required")]

        try:
            file_path = get_db_file_path(key)

            if not file_path.exists():
                return [
                    types.TextContent(
                        type="text", text=f"Key '{key}' not found in database"
                    )
                ]

            with open(file_path, "r") as f:
                data = json.load(f)

            return [
                types.TextContent(
                    type="text",
                    text=f"Key: {data['key']}\nValue: {data['value']}\nStored at: {data['timestamp']}",
                )
            ]

        except Exception as e:
            return [
                types.TextContent(
                    type="text", text=f"Error reading key '{key}': {str(e)}"
                )
            ]

    elif name == "gettime":
        try:
            now = datetime.now()
            utc_now = datetime.utcnow()

            local_time = now.strftime("%Y-%m-%d %H:%M:%S %Z")
            utc_time = utc_now.strftime("%Y-%m-%d %H:%M:%S UTC")

            return [
                types.TextContent(
                    type="text", text=f"Local time: {local_time}\nUTC time: {utc_time}"
                )
            ]

        except Exception as e:
            return [
                types.TextContent(type="text", text=f"Error getting time: {str(e)}")
            ]

    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


class NotificationOptions:
    """Notification options for MCP server"""

    def __init__(self):
        self.tools_changed = False
        self.resources_changed = False
        self.prompts_changed = False


async def main():
    """Main entry point for the server"""
    # Run the server using stdio transport
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mynote",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
