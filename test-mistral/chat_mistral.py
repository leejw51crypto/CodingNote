import os
import requests
import json

def chat_with_mistral(message):
    # Get API key from environment variable
    api_key = os.getenv('MISTRAL_API_KEY')
    if not api_key:
        raise ValueError("Please set MISTRAL_API_KEY environment variable")

    # API endpoint
    url = "https://api.mistral.ai/v1/chat/completions"

    # Headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Request body
    data = {
        "model": "mistral-small-latest",  # Using the small model
        "messages": [
            {
                "role": "user",
                "content": message
            }
        ]
    }

    try:
        # Make the API call
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Parse and return the response
        result = response.json()
        return result['choices'][0]['message']['content']

    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None

def main():
    # Simple hello world message
    message = "Hello! Can you introduce yourself?"
    
    print("Sending message to Mistral AI...")
    response = chat_with_mistral(message)
    
    if response:
        print("\nMistral AI response:")
        print(response)

if __name__ == "__main__":
    main() 