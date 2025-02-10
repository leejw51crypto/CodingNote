import os
import json
import functools
import random
from typing import Dict, List
import requests
import re

# Sample data and helper functions remain the same
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

# Define the function schemas for Ollama
function_schemas = [
    {
        "name": "retrieve_payment_status",
        "description": "Get payment status of a transaction id",
        "parameters": {
            "type": "object",
            "required": ["transaction_id"],
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "The transaction id.",
                }
            },
        },
    },
    {
        "name": "retrieve_payment_date",
        "description": "Get payment date of a transaction id",
        "parameters": {
            "type": "object",
            "required": ["transaction_id"],
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "The transaction id.",
                }
            },
        },
    },
    {
        "name": "calculate_fibonacci",
        "description": "Calculate the nth Fibonacci number",
        "parameters": {
            "type": "object",
            "required": ["n"],
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "The position in the Fibonacci sequence (0-based index)",
                }
            },
        },
    },
    {
        "name": "get_weather",
        "description": "Get current weather information for a city",
        "parameters": {
            "type": "object",
            "required": ["city"],
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city to get weather for",
                }
            },
        },
    }
]

def call_ollama(messages, function_schemas):
    url = "http://localhost:11434/api/chat"
    
    # Convert messages to the format Ollama expects
    formatted_messages = []
    for msg in messages:
        formatted_msg = {
            "role": msg["role"],
            "content": msg["content"]
        }
        if msg["role"] == "tool":
            formatted_msg["name"] = msg["name"]
        formatted_messages.append(formatted_msg)
    
    payload = {
        "model": "mistral",
        "messages": formatted_messages,
        "stream": False,
        "format": "json"  # Request JSON response
    }
    
    try:
        print("Sending request to Ollama...")
        response = requests.post(url, json=payload)
        response.raise_for_status()
        response_data = response.json()
        print(f"Response from Ollama: {json.dumps(response_data, indent=2)}")
        return response_data
    except requests.exceptions.RequestException as e:
        error_msg = f"Failed to call Ollama API: {str(e)}"
        print(f"Error: {error_msg}")
        return {"error": error_msg}
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse Ollama response: {str(e)}"
        print(f"Error: {error_msg}")
        return {"error": error_msg}

def main():
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

        # First, try to handle the query directly with our functions
        if "status" in user_input.lower() and "T" in user_input:
            # Extract transaction ID (assuming format like T1001)
            transaction_id = re.search(r'T\d+', user_input)
            if transaction_id:
                transaction_id = transaction_id.group(0)
                result = json.loads(retrieve_payment_status(data, transaction_id))
                print(f"Assistant: The status of transaction {transaction_id} is {result['status']}")
                continue

        messages.append({"role": "user", "content": user_input})
        response = call_ollama(messages, function_schemas)
        
        if "error" in response:
            print(f"Error: {response['error']}")
            continue
            
        # Extract the assistant's response
        assistant_content = response.get("message", {}).get("content", "")
        if not assistant_content and "response" in response:
            assistant_content = response["response"]
        
        messages.append({"role": "assistant", "content": assistant_content})
        print(f"Assistant: {assistant_content}")
        print()

if __name__ == "__main__":
    main() 