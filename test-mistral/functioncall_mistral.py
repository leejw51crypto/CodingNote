import os
import json
import functools
import random
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from mistralai.client import MistralClient
from mistralai.models.chat_completion import (
    ChatMessage,
    Function,
    ToolType,
    ToolCall,
    ToolChoice
)

# Create sample data
data = {
    'transaction_id': ['T1001', 'T1002', 'T1003', 'T1004', 'T1005'],
    'customer_id': ['C001', 'C002', 'C003', 'C002', 'C001'],
    'payment_amount': [125.50, 89.99, 120.00, 54.30, 210.20],
    'payment_date': ['2021-10-05', '2021-10-06', '2021-10-07', '2021-10-05', '2021-10-08'],
    'payment_status': ['Paid', 'Unpaid', 'Paid', 'Paid', 'Pending']
}

def retrieve_payment_status(data: Dict[str, List], transaction_id: str) -> str:
    for i, r in enumerate(data["transaction_id"]):
        if r == transaction_id:
            return json.dumps({"status": data["payment_status"][i]})
    return json.dumps({"status": "Error - transaction id not found"})

def retrieve_payment_date(data: Dict[str, List], transaction_id: str) -> str:
    for i, r in enumerate(data["transaction_id"]):
        if r == transaction_id:
            return json.dumps({"date": data["payment_date"][i]})
    return json.dumps({"status": "Error - transaction id not found"})

def calculate_fibonacci(n: int) -> str:
    def fib_sequence(n: int) -> list:
        if n <= 0:
            return [0]
        elif n == 1:
            return [0, 1]
        
        sequence = [0, 1]
        for _ in range(2, n + 1):
            sequence.append(sequence[-1] + sequence[-2])
        return sequence
    
    try:
        n = int(n)
        if n < 0:
            return json.dumps({"error": "Please provide a non-negative number"})
        sequence = fib_sequence(n)
        return json.dumps({
            "sequence": sequence,
            "result": sequence[-1],
            "length": len(sequence)
        })
    except ValueError:
        return json.dumps({"error": "Please provide a valid integer"})

def get_weather(city: str) -> str:
    # List of possible weather descriptions
    descriptions = [
        "clear sky",
        "few clouds",
        "scattered clouds",
        "broken clouds",
        "light rain",
        "moderate rain",
        "sunny",
        "partly cloudy",
        "overcast"
    ]
    
    # Generate random weather data
    result = {
        "temperature": round(random.uniform(-5, 35), 1),  # Temperature between -5°C and 35°C
        "humidity": random.randint(30, 90),  # Humidity between 30% and 90%
        "description": random.choice(descriptions),
        "wind_speed": round(random.uniform(0, 30), 1)  # Wind speed between 0 and 30 m/s
    }
    return json.dumps(result)

names_to_functions = {
    "retrieve_payment_status": functools.partial(retrieve_payment_status, data=data),
    "retrieve_payment_date": functools.partial(retrieve_payment_date, data=data),
    "calculate_fibonacci": calculate_fibonacci,
    "get_weather": get_weather,
}

tools = [
    {
        "type": "function",
        "function": Function(
            name="retrieve_payment_status",
            description="Get payment status of a transaction id",
            parameters={
                "type": "object",
                "required": ["transaction_id"],
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction id.",
                    }
                },
            },
        ),
    },
    {
        "type": "function",
        "function": Function(
            name="retrieve_payment_date",
            description="Get payment date of a transaction id",
            parameters={
                "type": "object",
                "required": ["transaction_id"],
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction id.",
                    }
                },
            },
        ),
    },
    {
        "type": "function",
        "function": Function(
            name="calculate_fibonacci",
            description="Calculate the nth Fibonacci number",
            parameters={
                "type": "object",
                "required": ["n"],
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "The position in the Fibonacci sequence (0-based index)",
                    }
                },
            },
        ),
    },
    {
        "type": "function",
        "function": Function(
            name="get_weather",
            description="Get current weather information for a city",
            parameters={
                "type": "object",
                "required": ["city"],
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city to get weather for",
                    }
                },
            },
        ),
    },
]

def main():
    api_key = os.environ["MISTRAL_API_KEY"]
    model = "mistral-small-latest"
    client = MistralClient(api_key=api_key)
    messages = []
    
    print("Welcome to the interactive chat! (Type 'quit' to exit)")
    print("You can:")
    print("1. Ask about transaction status (e.g., 'What's the status of transaction T1001?')")
    print("2. Ask about transaction date (e.g., 'When was transaction T1002 made?')")
    print("3. Calculate Fibonacci numbers (e.g., 'What is the 10th Fibonacci number?')")
    print("4. Get weather information (e.g., 'What's the weather in London?')")
    print()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        messages.append(ChatMessage(role="user", content=user_input))
        response = client.chat(model=model, messages=messages, tools=tools)
        assistant_message = response.choices[0].message
        messages.append(assistant_message)
        
        # Handle tool calls if any
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_params = json.loads(tool_call.function.arguments)
                print(f"Assistant is using {function_name}...")
                function_result = names_to_functions[function_name](**function_params)
                messages.append(
                    ChatMessage(
                        role="tool",
                        name=function_name,
                        content=function_result,
                        tool_call_id=tool_call.id,
                    )
                )
            
            # Get final response after tool calls
            response = client.chat(model=model, messages=messages, tools=tools)
            assistant_message = response.choices[0].message
            messages.append(assistant_message)

        print(f"Assistant: {assistant_message.content}")
        print()

if __name__ == "__main__":
    main()