import requests
import json
from typing import Dict, Any, List, Optional


def get_current_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
    """
    Get the current weather in a given location.
    This is a mock function for demonstration purposes.
    """
    return {
        "location": location,
        "temperature": "22",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }


def calculate_fibonacci(n: int) -> Dict[str, Any]:
    """
    Calculate the Fibonacci sequence up to n numbers.
    """
    if n <= 0:
        return {"error": "Number must be positive"}

    fib = [0, 1]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])

    return {"sequence": fib, "length": len(fib), "last_number": fib[-1]}


def extract_function_call(response_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract function call from potentially nested JSON responses
    """
    if not isinstance(response_obj, dict):
        return None
    
    # Check if this is a direct function call
    if "choices" in response_obj and len(response_obj["choices"]) > 0:
        message = response_obj["choices"][0].get("message", {})
        if "function_call" in message:
            return message["function_call"]
    
    # Check if this is a nested JSON string in content
    if "choices" in response_obj and len(response_obj["choices"]) > 0:
        message = response_obj["choices"][0].get("message", {})
        content = message.get("content", "")
        if content and isinstance(content, str):
            # Try to parse content as JSON if it looks like JSON
            content = content.strip()
            if content.startswith("{") and content.endswith("}"):
                try:
                    nested_json = json.loads(content)
                    return extract_function_call(nested_json)
                except json.JSONDecodeError:
                    pass
            # Handle case where JSON is wrapped in markdown code blocks
            elif "```json" in content or "```" in content:
                try:
                    # Extract JSON from markdown code blocks
                    json_content = content.split("```")[1]
                    if json_content.startswith("json"):
                        json_content = json_content[4:]
                    nested_json = json.loads(json_content.strip())
                    return extract_function_call(nested_json)
                except (json.JSONDecodeError, IndexError):
                    pass
    
    return None


def query_ollama_with_functions(
    prompt: str, functions: List[Dict[str, Any]], function_call: str = "auto"
) -> Dict[str, Any]:
    """
    Send a query to the Ollama API with function definitions
    """
    url = "http://localhost:11434/api/generate"

    # Construct the prompt with function definitions in OpenAI format
    system_prompt = """You are a helpful assistant that can call functions.
    When a user asks about weather or calculations, you MUST call the appropriate function.
    
    IMPORTANT: You must ALWAYS respond in one of these two JSON formats EXACTLY:

    1. For function calls (use this when user asks about weather or calculations):
    {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "deepseek-coder-v2",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "function_call": {
                    "name": "get_current_weather",
                    "arguments": "{\\"location\\": \\"New York\\", \\"unit\\": \\"celsius\\"}"
                }
            },
            "finish_reason": "function_call"
        }]
    }

    2. For regular responses (use this when no function is needed):
    {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "deepseek-coder-v2",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Your response here"
            },
            "finish_reason": "stop"
        }]
    }

    EXAMPLES:
    User: What's the weather in Tokyo?
    Assistant: {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "deepseek-coder-v2",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "function_call": {
                    "name": "get_current_weather",
                    "arguments": "{\\"location\\": \\"Tokyo\\", \\"unit\\": \\"celsius\\"}"
                }
            },
            "finish_reason": "function_call"
        }]
    }

    User: Calculate 5 Fibonacci numbers
    Assistant: {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "deepseek-coder-v2",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "function_call": {
                    "name": "calculate_fibonacci",
                    "arguments": "{\\"n\\": 5}"
                }
            },
            "finish_reason": "function_call"
        }]
    }

    Available functions:
    """

    # Add function definitions to system prompt
    system_prompt += json.dumps(functions, indent=2)

    # Combine system prompt with user prompt
    full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"

    data = {
        "model": "deepseek-coder-v2",
        "prompt": full_prompt,
        "stream": False,
    }

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()

        # Try to parse the response as JSON
        try:
            parsed_response = json.loads(result["response"])
            return parsed_response
        except json.JSONDecodeError:
            # If not JSON, wrap the response in OpenAI format
            return {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "deepseek-coder-v2",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result["response"]},
                        "finish_reason": "stop",
                    }
                ],
            }

    except Exception as e:
        return {"error": str(e)}


def main():
    # Define available functions
    functions = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use",
                    },
                },
                "required": ["location"],
            },
        },
        {
            "name": "calculate_fibonacci",
            "description": "Calculate Fibonacci sequence up to n numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of Fibonacci numbers to generate (must be positive)",
                    }
                },
                "required": ["n"],
            },
        },
    ]

    print("Interactive AI Assistant")
    print("Available commands:")
    print("1. Ask about weather (e.g., 'What's the weather like in Seoul?')")
    print("2. Calculate Fibonacci (e.g., 'Calculate first 10 Fibonacci numbers')")
    print("3. Type 'quit' to exit")
    print("\nWhat would you like to know?\n")

    while True:
        try:
            # Get user input
            prompt = input("> ")

            # Check for quit command
            if prompt.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            print("\nSending query to model...")

            # Get the model's response
            response = query_ollama_with_functions(prompt, functions)
            print(f"Initial response:\n{json.dumps(response, indent=2)}\n")

            # Try to extract function call from potentially nested response
            function_call = extract_function_call(response)
            
            if function_call:
                function_name = function_call["name"]
                # Parse the arguments string into a dictionary
                arguments = json.loads(function_call["arguments"])

                # Execute the appropriate function
                result = None
                if function_name == "get_current_weather":
                    result = get_current_weather(**arguments)
                elif function_name == "calculate_fibonacci":
                    result = calculate_fibonacci(**arguments)

                if result:
                    print(f"Function result:\n{json.dumps(result, indent=2)}")

                    # Send the result back to the model for a natural language response
                    second_prompt = f"Here's the {function_name} result: {json.dumps(result)}. Can you summarize this in a natural way?"
                    final_response = query_ollama_with_functions(
                        second_prompt, functions
                    )
                    print(
                        f"\nFinal response:\n{json.dumps(final_response, indent=2)}"
                    )
            else:
                # Check if there's a regular content response
                if "choices" in response and len(response["choices"]) > 0:
                    message = response["choices"][0].get("message", {})
                    content = message.get("content")
                    if content:
                        print("Response:", content)
                    else:
                        print("No valid response content found.")
                else:
                    print("No valid response format detected.")

            print("\nWhat else would you like to know?\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try again.\n")


if __name__ == "__main__":
    main()
