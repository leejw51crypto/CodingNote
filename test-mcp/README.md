# MyNote MCP Server

## ⚠️ IMPORTANT: Use Absolute Paths Only
**Never use relative paths in your MCP configuration!** Always use full absolute paths or `$HOME` references.

A simple Model Context Protocol (MCP) server that provides a key-value database with file storage for taking and managing notes.

## What is MCP?

Model Context Protocol (MCP) is an open standard that enables AI assistants to securely connect with external data sources and tools. This MyNote server implements MCP to provide:

- **Tools**: Write and read notes, get current time
- **Resources**: Access database statistics and stored notes
- **Prompts**: Guided workflows for creating, querying, and organizing notes

## Setup Instructions

### 1. Locate Your Project Folder

First, determine the **absolute path** to where you've placed this MyNote server:

```bash
# Navigate to your project directory
cd /path/to/your/mynote/project

# Get the absolute path (copy this for configuration)mcp-server-mynote.log
pwd
```

**Example paths:**
- `$HOME/Documents/mynote-server/`
- `$HOME/projects/mynote-mcp/`
- `/Users/username/projects/mynote/`

### 2. Install Dependencies

Make sure you have Python 3.8+ installed. Install the required MCP package:

```bash
pip install mcp
```

### 3. Configuration

#### **Config** Location
The Claude Desktop configuration file is located at:
```
$HOME/Library/Application Support/Claude/claude_desktop_config.json
```

#### Configuration Example
Add the MyNote server to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mynote": {
      "command": "/usr/bin/python3",
      "args": [
        "/FULL/ABSOLUTE/PATH/TO/mynote_server.py"
      ],
      "env": {
        "PYTHONPATH": "/FULL/ABSOLUTE/PATH/TO/PROJECT"
      }
    }
  }
}
```

**Replace with your actual paths:**
```json
{
  "mcpServers": {
    "mynote": {
      "command": "/opt/anaconda3/bin/python",
      "args": [
        "$HOME/your-project-path/mynote_server.py"
      ],
      "env": {
        "PYTHONPATH": "$HOME/your-project-path"
      }
    }
  }
}
```

**Critical Requirements**: 
- **NEVER use relative paths** like `./mynote_server.py` or `../test-mcp/`
- Always use **full absolute paths** starting from root (`/`) or `$HOME`
- Use the exact path from step 1 above
- Find your Python path with: `which python3`

### 4. Make Server Executable

```bash
chmod +x mynote_server.py
```

### 5. Restart Claude Desktop

After updating the configuration, restart Claude Desktop to load the MCP server.

## Features

### Tools Available

1. **write**: Store a key-value pair in the database
   - Parameters: `key` (string), `value` (string)
   - Example: Store meeting notes, ideas, or any text content

2. **read**: Retrieve a value by its key from the database
   - Parameters: `key` (string)
   - Returns the stored value with timestamp

3. **gettime**: Get current local and UTC time
   - No parameters required
   - Useful for timestamping notes

### Resources Available

1. **mynote://stats**: Database statistics including total keys, size, and key list
2. **mynote://keys**: List of all stored keys
3. **mynote://key/{key}**: Individual note content for any stored key

### Prompts Available

1. **create_note**: Guided workflow for creating well-structured notes
2. **query_notes**: Help search and find relevant notes
3. **organize_notes**: Assistance with organizing and categorizing notes

## Usage Examples

### Basic Note Taking

```
You: Please store a note about today's meeting
Assistant: I'll help you create a note about today's meeting. What were the main topics discussed?

You: We discussed the Q4 budget and new hiring plans
Assistant: [Uses write tool to store the note with a proper key like "meeting_2024_01_15_q4_budget"]
```

### Searching Notes

```
You: Find all notes related to budget
Assistant: [Uses query_notes prompt and read tool to search through stored notes]
```

### Getting Database Statistics

```
You: Show me my note database stats
Assistant: [Accesses mynote://stats resource to show total notes, size, etc.]
```

## Data Storage

- Notes are stored in `$HOME/.mynote_db/` directory
- Each note is saved as a JSON file with:
  - Original key
  - Content value
  - Creation timestamp
- File names are sanitized for filesystem compatibility

## Logs and Debugging

### Log Location
MCP server logs are stored at:
```
$HOME/Library/Logs/Claude/mcp-server-mynote.log
```

Check this file for error messages and debugging information when the server isn't working properly.

### Viewing Logs
```bash
# View recent log entries
tail -f "$HOME/Library/Logs/Claude/mcp-server-mynote.log"

# View all logs
cat "$HOME/Library/Logs/Claude/mcp-server-mynote.log"
```

## Troubleshooting

### Server Not Loading

1. **Check Python Path**: Ensure the `command` path in config points to your Python executable
2. **Check File Path**: Verify the path to `mynote_server.py` is correct (use absolute paths, not relative)
3. **Check Permissions**: Make sure the server file is executable (`chmod +x mynote_server.py`)
4. **Check Dependencies**: Ensure `mcp` package is installed (`pip install mcp`)
5. **Check Logs**: Review the log file at `$HOME/Library/Logs/Claude/mcp-server-mynote.log` for error details

### Permission Issues

If you get permission errors:
```bash
# Make sure the database directory is writable
mkdir -p $HOME/.mynote_db
chmod 755 $HOME/.mynote_db
```

### Claude Desktop Not Recognizing Server

1. Restart Claude Desktop completely
2. Check the JSON syntax in `claude_desktop_config.json`
3. Look for any syntax errors or missing commas
4. Verify the server starts without errors: `python3 mynote_server.py`
5. Check the MCP server logs for specific error messages
6. Ensure all paths in the configuration are absolute (not relative)

### Debug Mode

To test the server independently:
```bash
python3 mynote_server.py
# The server should start and wait for MCP messages
```

## Advanced Configuration
### Custom Database Location

You can modify the database location by editing `mynote_server.py`:
```python
# Change **this** line to use a custom location
DB_DIR = Path.home() / ".mynote_db"
```

### Environment Variables

You can use environment variables in your configuration:
```json
{
  "mcpServers": {
    "mynote": {
      "command": "python3",
      "args": ["$HOME/path/to/mynote_server.py"],
      "env": {
        "PYTHONPATH": "$HOME/path/to/directory",
        "MYNOTE_DB_PATH": "$HOME/custom_notes"
      }
    }
  }
}
```

## Security Notes

- All data is stored locally in your home directory
- No network connections are made by the server
- File access is restricted to the designated database directory
- Keys are sanitized to prevent directory traversal attacks