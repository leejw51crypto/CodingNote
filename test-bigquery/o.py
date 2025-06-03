#!/usr/bin/env python3
"""
OpenAI Hello World Example
Uses OPENAI_API_KEY environment variable for authentication
"""

import os
from openai import OpenAI


def main():
    # Initialize OpenAI client with API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        return

    # Create OpenAI client
    client = OpenAI(api_key=api_key)

    try:
        # Make a simple chat completion request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello world in a creative way!"},
            ],
            max_tokens=100,
            temperature=0.7,
        )

        # Extract and print the response
        message = response.choices[0].message.content
        print("OpenAI Response:")
        print("-" * 40)
        print(message)
        print("-" * 40)

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")


if __name__ == "__main__":
    main()
