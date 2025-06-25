import os
import json
from datetime import datetime
from mistralai import Mistral

# Initialize the client
api_key = os.environ.get("MISTRAL_API_KEY")
client = Mistral(api_key)

# Initialize conversation history
messages = []


# Define example functions
def get_current_time():
    """Get the current time."""
    return {"current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    return {"result": a + b, "operation": "addition"}


def get_weather_mock(location):
    """Get mock weather information for a location."""
    return {
        "location": location,
        "temperature": "22°C",
        "condition": "Sunny",
        "humidity": "60%",
    }


# Define tools for function calling (OpenAI-style)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_sum",
            "description": "Calculate the sum of two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_mock",
            "description": "Get weather information for a location (mock data)",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get weather for",
                    }
                },
                "required": ["location"],
            },
        },
    },
]

# Function registry
FUNCTION_REGISTRY = {
    "get_current_time": get_current_time,
    "calculate_sum": calculate_sum,
    "get_weather_mock": get_weather_mock,
}


def execute_function(function_name, arguments):
    """Execute a function call with given arguments."""
    if function_name not in FUNCTION_REGISTRY:
        return {"error": f"Function {function_name} not found"}

    try:
        func = FUNCTION_REGISTRY[function_name]
        if arguments:
            return func(**arguments)
        else:
            return func()
    except Exception as e:
        return {"error": f"Error executing {function_name}: {str(e)}"}


def handle_function_calls(response):
    """Handle function calls from the model response."""
    if not response.choices[0].message.tool_calls:
        return None

    tool_calls = response.choices[0].message.tool_calls
    function_results = []

    print("\n🔧 Function calls detected:")
    print("=" * 40)

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        arguments = (
            json.loads(tool_call.function.arguments)
            if tool_call.function.arguments
            else {}
        )

        print(f"🔧 Executing function: {function_name}")
        print(f"📋 Arguments: {json.dumps(arguments, indent=2)}")

        # Execute the function
        result = execute_function(function_name, arguments)

        print(f"📤 Result: {json.dumps(result, indent=2)}")
        print("-" * 40)

        # Prepare function result for the model
        function_results.append(
            {
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id,
                "name": function_name,
            }
        )

    return function_results


def main():
    print("🤖 Mistral Chat - Interactive Mode with Function Calling")
    print("Available functions:")
    print("  • get_current_time - Get current date and time")
    print("  • calculate_sum - Add two numbers")
    print("  • get_weather_mock - Get mock weather data")
    print("\nType 'quit', 'exit', or 'bye' to end the conversation")
    print("Type 'json' to toggle JSON output mode")
    print("Type 'clear' to clear conversation history")
    print("-" * 60)

    json_mode = False

    while True:
        # Get user input
        try:
            user_input = input("\n💬 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        # Check for special commands
        if user_input.lower() in ["quit", "exit", "bye", "q"]:
            print("👋 Goodbye!")
            break

        if user_input.lower() == "json":
            json_mode = not json_mode
            print(f"📋 JSON output mode: {'ON' if json_mode else 'OFF'}")
            continue

        if user_input.lower() == "clear":
            messages.clear()
            print("🗑️ Conversation history cleared!")
            continue

        # Skip empty inputs
        if not user_input:
            continue

        # Add user message to conversation history
        messages.append({"role": "user", "content": user_input})

        try:
            print("\n🤖 Mistral: ", end="", flush=True)

            # Make the API call with tools (OpenAI-style)
            response = client.chat.complete(
                # model="mistral-large-latest",
                model="magistral-medium-latest",
                messages=messages,
                tools=tools,
                tool_choice="auto",  # Let the model decide when to use tools
                temperature=0.7,
                max_tokens=2048,
            )

            # Get the assistant's response
            assistant_message = response.choices[0].message

            # Check if there are function calls
            if assistant_message.tool_calls:
                # Add the assistant's message (with tool calls) to conversation
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in assistant_message.tool_calls
                        ],
                    }
                )

                # Handle function calls
                function_results = handle_function_calls(response)

                if function_results:
                    # Add function results to conversation
                    messages.extend(function_results)

                    # Make another call to get the final response
                    print("🤖 Processing function results...")

                    final_response = client.chat.complete(
                        model="mistral-large-latest",
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        temperature=0.7,
                        max_tokens=2048,
                    )

                    final_message = final_response.choices[0].message
                    print(f"🤖 Final response: {final_message.content}")

                    # Add final response to conversation
                    messages.append(
                        {"role": "assistant", "content": final_message.content}
                    )
            else:
                # Regular response without function calls
                print(assistant_message.content)
                messages.append(
                    {"role": "assistant", "content": assistant_message.content}
                )

            # Output JSON if requested
            if json_mode:
                print("\n📋 JSON Output:")
                print(
                    json.dumps(
                        {
                            "conversation_length": len(messages),
                            "last_exchange": {
                                "user": user_input,
                                "assistant": (
                                    messages[-1]["content"] if messages else ""
                                ),
                                "function_calls_made": (
                                    len(assistant_message.tool_calls)
                                    if assistant_message.tool_calls
                                    else 0
                                ),
                            },
                        },
                        indent=2,
                    )
                )

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'quit' to exit.")


if __name__ == "__main__":
    main()
