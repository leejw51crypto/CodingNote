import requests
import json
import sys
import re
import readline  # For input history in terminal


def get_weather(city: str) -> str:
    """Get weather information for a given city"""
    try:
        url = f"https://wttr.in/{city}?format=%C+%t"
        response = requests.get(url)
        return response.text.strip()
    except Exception as e:
        return f"Error getting weather: {str(e)}"


def fibonacci(n: int) -> list:
    """Calculate the Fibonacci sequence up to the nth number"""
    print(f"Calculating Fibonacci sequence up to {n}")
    if n <= 0:
        return [0]
    elif n == 1:
        return [0, 1]
    else:
        fib_list = [0, 1]
        for i in range(2, n + 1):
            fib_list.append(fib_list[i - 1] + fib_list[i - 2])
        return fib_list


# Define function schemas in OpenAI-style format
FUNCTION_DEFINITIONS = [
    {
        "name": "get_weather",
        "description": "Get weather information for a given city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city to get weather for"}
            },
            "required": ["city"],
        },
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


def execute_function_call(function_name: str, arguments: dict) -> str:
    """Execute a function call based on name and arguments"""
    print(f"Function name: {function_name}")
    print(f"Arguments: {arguments}")

    # Normalize function name
    normalized_name = function_name.lower()

    if "weather" in normalized_name:
        # Handle different parameter names for city
        city = None
        for key in ["city", "location", "place", "town"]:
            if key in arguments:
                city = arguments.get(key, "")
                break

        if not city:
            return "Error: No city specified for weather"

        result = get_weather(city)
        return f"Weather in {city}: {result}"
    elif "fibonacci" in normalized_name:
        try:
            # Handle different parameter names for n
            n = None
            for key in ["n", "number", "position", "index"]:
                if key in arguments:
                    n = int(arguments.get(key, 0))
                    break

            if n is None:
                return "Error: No number specified for Fibonacci"

            print(f"Calling fibonacci function with n={n}")
            result = fibonacci(n)
            print(f"Fibonacci result: {result}")
            sequence_str = ", ".join(map(str, result))
            return f"Fibonacci sequence up to {n}: {sequence_str}"
        except ValueError as e:
            print(f"ValueError in fibonacci: {e}")
            return "Error: Invalid number for Fibonacci calculation"
        except Exception as e:
            print(f"Unexpected error in fibonacci: {e}")
            return f"Error calculating Fibonacci: {str(e)}"
    return f"Error: Unknown function '{function_name}'"


def generate_response(
    prompt: str, model_name="llama3", with_functions=True, request_json=False
):
    """
    Generate a response using the Ollama API

    Args:
        prompt (str): The input prompt for the model
        model_name (str): The name of the Ollama model to use
        with_functions (bool): Whether to include function definitions
        request_json (bool): Whether to request JSON response format
    """
    url = "http://localhost:11434/api/generate"

    # Modify prompt to request JSON response if needed
    if request_json:
        prompt = (
            f"{prompt}\n\nPlease respond with a JSON object in the following format:\n"
            + "```json\n"
            + "{\n"
            + '  "name": "function_name",\n'
            + '  "arguments": {\n'
            + '    "param1": "value1",\n'
            + '    "param2": "value2"\n'
            + "  }\n"
            + "}\n"
            + "```\n"
            + "If no function call is needed, respond with null for the name.\n"
            + "For weather, use get_weather function with city parameter.\n"
            + "For fibonacci, use fibonacci function with n parameter."
        )

    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 1024},
    }

    # Add function definitions if requested
    if with_functions:
        # Ollama supports OpenAI-style function calling
        data["functions"] = FUNCTION_DEFINITIONS

        # Some models support tool_choice to force function calling
        if request_json:
            data["tool_choice"] = "auto"

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {str(e)}")
        return None


def parse_function_call_json(response_text: str) -> tuple:
    """Parse function call from JSON response"""
    try:
        # Check if the response is already in the Ollama function call format
        if "tool_calls" in response_text:
            try:
                response_data = json.loads(response_text)
                tool_calls = response_data.get("tool_calls", [])
                if tool_calls and len(tool_calls) > 0:
                    function_call = tool_calls[0].get("function", {})
                    function_name = function_call.get("name")
                    arguments = json.loads(function_call.get("arguments", "{}"))
                    return function_name, arguments
            except json.JSONDecodeError:
                # Not valid JSON, continue with regex parsing
                pass

        # Debug: Print the first 200 characters of the response
        print(f"DEBUG: Response text (first 200 chars): {response_text[:200]}")

        # First, check if the response contains "null" for the name
        if (
            '"name"\\s*:\\s*"?null"?' in response_text
            or '"name"\\s*:\\s*null' in response_text
        ):
            return None, None

        # Direct pattern match for the exact format seen in examples
        exact_pattern = r'{\s*"name"\s*:\s*"fibonacci"\s*,\s*"arguments"\s*:\s*{\s*"n"\s*:\s*(\d+)\s*}\s*}'
        exact_match = re.search(exact_pattern, response_text)
        if exact_match:
            print(f"DEBUG: Found exact Fibonacci pattern match")
            n = int(exact_match.group(1))
            return "fibonacci", {"n": n}

        # Preprocess: Remove comments from JSON (both // and /* */ style)
        def remove_comments(json_str):
            # Remove // comments
            json_str = re.sub(r"//.*?(\n|$)", r"\1", json_str)
            # Remove /* */ comments
            json_str = re.sub(r"/\*.*?\*/", "", json_str, flags=re.DOTALL)
            return json_str

        # Look for function call pattern in the response
        function_call_pattern = r"```json\s*({[\s\S]*?})\s*```"
        match = re.search(function_call_pattern, response_text)

        if match:
            function_json = match.group(1)
            # Clean JSON before parsing
            function_json = remove_comments(function_json)
            try:
                function_data = json.loads(function_json)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                # Try a more lenient approach - strip trailing commas
                function_json = re.sub(r",\s*}", "}", function_json)
                function_json = re.sub(r",\s*]", "]", function_json)
                try:
                    function_data = json.loads(function_json)
                except json.JSONDecodeError:
                    # If still failing, try to extract just the essential parts
                    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', function_json)
                    args_match = re.search(
                        r'"arguments"\s*:\s*({[^}]+})', function_json
                    )

                    if name_match and args_match:
                        function_name = name_match.group(1)
                        try:
                            arguments = json.loads(args_match.group(1))
                        except:
                            # Create a simple dict with city if we can find it
                            city_match = re.search(
                                r'"city"\s*:\s*"([^"]+)"', args_match.group(1)
                            )
                            if city_match:
                                arguments = {"city": city_match.group(1)}
                            else:
                                arguments = {}
                        return function_name, arguments
                    else:
                        return None, None

            function_name = function_data.get("name")
            if not function_name or function_name == "null":
                return None, None

            # Normalize function name by handling common variations
            if "weather" in function_name.lower():
                function_name = "get_weather"
            elif "fibonacci" in function_name.lower():
                function_name = "fibonacci"

            arguments = function_data.get("arguments", {})

            # If arguments is a string, try to parse it as JSON
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except:
                    # If parsing fails, create a simple dict with the string
                    arguments = {"value": arguments}

            return function_name, arguments

        # Try alternative pattern without code blocks
        alt_pattern = (
            r'[\{\[].*"name"\s*:\s*"([^"]+)".*"arguments"\s*:\s*(\{[^\}]+\})[\}\]]'
        )
        alt_match = re.search(alt_pattern, response_text)
        if alt_match:
            function_name = alt_match.group(1)
            arguments_str = alt_match.group(2)

            # Normalize function name by handling common variations
            if "weather" in function_name.lower():
                function_name = "get_weather"
            elif "fibonacci" in function_name.lower():
                function_name = "fibonacci"

            try:
                # Clean JSON before parsing
                arguments_str = remove_comments(arguments_str)
                arguments = json.loads(arguments_str)
            except:
                arguments = {}

            return function_name, arguments

        # Try to find any mention of weather or fibonacci in the text
        if re.search(
            r'weather.*(city|location|place).*["\']([^"\']+)["\']',
            response_text,
            re.IGNORECASE,
        ):
            match = re.search(
                r'weather.*(city|location|place).*["\']([^"\']+)["\']',
                response_text,
                re.IGNORECASE,
            )
            param_name = match.group(1).lower()
            city = match.group(2)
            return "get_weather", {param_name: city}

        if re.search(
            r"fibonacci.*(number|position|n|index).*?(\d+)",
            response_text,
            re.IGNORECASE,
        ):
            match = re.search(
                r"fibonacci.*(number|position|n|index).*?(\d+)",
                response_text,
                re.IGNORECASE,
            )
            param_name = match.group(1).lower()
            n = int(match.group(2))
            return "fibonacci", {param_name: n}

        return None, None
    except Exception as e:
        print(f"Error parsing function call: {str(e)}")
        return None, None


def process_user_input(prompt: str, history: list, model_name="llama3"):
    """Process user input with function calling"""
    print("Response: ", end="", flush=True)

    # Call the model to check if we need to call a function
    response = generate_response(
        prompt, model_name=model_name, with_functions=True, request_json=True
    )

    if not response:
        print("Failed to get a response from the model.")
        return

    response_text = response.get("response", "")
    print("DEBUG: Got response from model")

    # Special handling for fibonacci queries
    if "fibonacci" in prompt.lower():
        print("DEBUG: Fibonacci query detected in prompt")
        print(f"DEBUG: Raw response: {response_text}")

    # Check if the response contains a function call
    function_name, arguments = parse_function_call_json(response_text)
    print(f"DEBUG: Parsed function call: name={function_name}, args={arguments}")

    result = ""
    if function_name:
        # Print the detected function call in a clean format
        print(f"Detected function call:")
        print(json.dumps({"name": function_name, "arguments": arguments}, indent=2))

        print(f"Calling function: {function_name}")
        function_result = execute_function_call(function_name, arguments)
        print(f"DEBUG: Function result: {function_result}")
        print(function_result)
        result = function_result
    else:
        print("DEBUG: No function call detected")
        # No function call needed, just print the response
        # Strip any JSON formatting instructions from the response
        clean_response = re.sub(
            r"Please respond with a JSON object.*", "", response_text, flags=re.DOTALL
        )
        clean_response = re.sub(
            r'```json\s*{\s*"name":\s*null\s*}\s*```',
            "",
            clean_response,
            flags=re.DOTALL,
        )
        print(clean_response.strip())
        result = clean_response.strip()

    # Add to history
    history.append({"input": prompt, "output": result})


def display_history(history: list):
    """Display chat history"""
    if not history:
        print("No chat history available.")
        return

    print("\n===== CHAT HISTORY =====")
    for i, entry in enumerate(history):
        print(f"[{i+1}] You: {entry['input']}")
        print(f"    Response: {entry['output']}")
    print("=======================\n")


def list_available_models():
    """List all available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            print("\nAvailable models:")
            for i, model in enumerate(models):
                print(f"{i+1}. {model.get('name')}")
            print()
            return models
        else:
            print("Failed to get models list")
            return []
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        return []


def main():
    print("Interactive Ollama Chat with Function Calling")

    # List available models
    models = list_available_models()

    # Default to llama3 if available, otherwise use the first model
    default_model = "llama3"
    if models:
        model_names = [model.get("name") for model in models]
        if default_model not in model_names and model_names:
            default_model = model_names[0]

    print(f"Using model: {default_model} (change with !model <name>)")
    model_name = default_model

    print("\nAvailable functions:")
    print("1. Weather information for a city")
    print("2. Calculate Fibonacci numbers")
    print("\nSpecial commands:")
    print("  !history - Show chat history")
    print("  !clear - Clear chat history")
    print("  !model <name> - Change the model")
    print("  !models - List available models")
    print("  !<number> - Repeat query from history")
    print("Type 'exit' to quit\n")

    # Initialize history
    history = []

    # Enable readline history
    readline.parse_and_bind("tab: complete")

    while True:
        try:
            prompt = input("You: ").strip()

            # Handle special commands
            if prompt.lower() == "exit":
                break
            elif not prompt:
                continue
            elif prompt == "!history":
                display_history(history)
                continue
            elif prompt == "!clear":
                history = []
                print("Chat history cleared.")
                continue
            elif prompt == "!models":
                list_available_models()
                continue
            elif prompt.startswith("!model "):
                new_model = prompt[7:].strip()
                model_name = new_model
                print(f"Model changed to: {model_name}")
                continue
            elif prompt.startswith("!") and prompt[1:].isdigit():
                # Repeat a query from history
                index = int(prompt[1:]) - 1
                if 0 <= index < len(history):
                    prompt = history[index]["input"]
                    print(f"Repeating: {prompt}")
                else:
                    print(
                        f"Invalid history index. Use !history to see available entries."
                    )
                    continue

            process_user_input(prompt, history, model_name=model_name)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
