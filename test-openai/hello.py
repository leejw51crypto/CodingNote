import os
import openai
from openai import OpenAI


def main():
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set the OPENAI_API_KEY environment variable")
        return

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    print("Welcome to OpenAI Chat! (Type 'quit' to exit)")

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        # Check for quit command
        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        try:
            # Create chat completion
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": user_input}],
            )

            # Print the response
            print("\nAssistant:", response.choices[0].message.content)

        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
