# ollama run llama3-groq-tool-use

import json
from datetime import datetime
import ollama


# Define our functions
def get_current_time():
    """Get the current time in ISO format"""
    return datetime.now().isoformat()


def fibonacci(n: int):
    """Calculate the nth Fibonacci number"""
    if n <= 0:
        return "Please provide a positive integer"
    if n == 1 or n == 2:
        return 1

    a, b = 1, 1
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b


# Define function schemas
function_schemas = [
    {
        "name": "get_current_time",
        "description": "Get the current time in ISO format",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "fibonacci",
        "description": "Calculate the nth Fibonacci number",
        "parameters": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "The position in the Fibonacci sequence to calculate",
                }
            },
            "required": ["n"],
        },
    },
]


def handle_function_call(function_call):
    """Handle the function call and return the result"""
    function_name = function_call.function.name
    arguments = function_call.function.arguments

    if function_name == "get_current_time":
        return get_current_time()
    elif function_name == "fibonacci":
        return fibonacci(arguments.get("n"))
    else:
        return "Function not found"


def main():
    # Initialize Ollama client
    client = ollama.Client()

    system_prompt = """You are a helpful AI assistant that can use functions.
When you need information about time, use the get_current_time function.
When you need to calculate Fibonacci numbers, use the fibonacci function.
Always use the available functions when appropriate.

Example function calls:
For time: {"name": "get_current_time", "arguments": "{}"}
For Fibonacci: {"name": "fibonacci", "arguments": {"n": 5}}

Remember to ALWAYS use these functions when asked about time or Fibonacci numbers."""

    print("Interactive Groq Function Call Test")
    print("Available functions:")
    print("1. get_current_time() - Get the current time")
    print("2. fibonacci(n) - Calculate the nth Fibonacci number")
    print("\nType 'exit' to quit")

    while True:
        try:
            print("\nUser: ", end="")
            prompt = input().strip()

            if prompt.lower() == "exit":
                print("Goodbye!")
                break

            if not prompt:
                continue

            response = client.chat(
                model="llama3-groq-tool-use",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                tools=function_schemas,
            )

            if hasattr(response.message, "tool_calls") and response.message.tool_calls:
                for tool_call in response.message.tool_calls:
                    result = handle_function_call(tool_call)
                    print(f"Function call: {tool_call.function.name}")
                    print(f"Result: {result}")

                    # Send the function result back to the model
                    follow_up = client.chat(
                        model="llama3-groq-tool-use",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                            {
                                "role": "assistant",
                                "content": "I'll help you with that.",
                                "tool_calls": [tool_call],
                            },
                            {"role": "tool", "content": str(result)},
                        ],
                        tools=function_schemas,
                    )
                    print(f"Assistant: {follow_up.message.content}")
            else:
                print(f"Assistant: {response.message.content}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            continue


if __name__ == "__main__":
    main()
