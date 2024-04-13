from openai import OpenAI
import json
from datetime import date
import math

client = OpenAI()

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

def get_current_date():
    """Get the current date"""
    today = date.today()
    return json.dumps({"date": today.strftime("%Y-%m-%d")})

def sin(degrees):
    """Calculate the sine of an angle in degrees"""
    radians = math.radians(degrees)
    return json.dumps({"sin": math.sin(radians)})

def run_conversation(prompt):
    # Step 1: send the conversation and available functions to the model
    messages = [{"role": "user", "content": prompt}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_date",
                "description": "Get the current date",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "sin",
                "description": "Calculate the sine of an angle in degrees",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "degrees": {
                            "type": "number",
                            "description": "The angle in degrees",
                        },
                    },
                    "required": ["degrees"],
                },
            },
        },
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_current_weather": get_current_weather,
            "get_current_date": get_current_date,
            "sin": sin,
        }  # added sin function
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            # print the function name and arguments
            print(f"Function name: {function_name}")
            print(f"Function arguments: {function_args}")
            
            function_response = function_to_call(**function_args)
            print(f"Function response: {function_response}")
            print("-----------------------------------")

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=messages,
        )  # get a new response from the model where it can see the function response
        return second_response

while True:
    user_prompt = input("Enter your prompt (or press Enter to exit): ")
    if not user_prompt:
        break
    response = run_conversation(user_prompt)
    print(response)
    # print all responses from the model
    for choice in response.choices:
        print(choice.message.content)