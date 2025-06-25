import os

from mistralai import Mistral

# Initialize the client
api_key = os.environ.get("MISTRAL_API_KEY")
client = Mistral(api_key)

# Initialize conversation history
conversation_history = []

# Define completion arguments
completion_args = {"temperature": 0.7, "max_tokens": 2048, "top_p": 1}

# Define tools if any
tools = []


def print_response(response):
    """Print the streaming response content."""
    full_response = ""
    for chunk in response:
        if hasattr(chunk, "data") and chunk.data:
            if hasattr(chunk.data, "content"):
                content = chunk.data.content
                print(content, end="", flush=True)
                full_response += content
    print()  # New line after response
    return full_response


def main():
    print("🤖 Mistral Chat - Interactive Mode")
    print("Type 'quit', 'exit', or 'bye' to end the conversation")
    print("-" * 50)

    while True:
        # Get user input
        try:
            user_input = input("\n💬 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        # Check for exit commands
        if user_input.lower() in ["quit", "exit", "bye", "q"]:
            print("👋 Goodbye!")
            break

        # Skip empty inputs
        if not user_input:
            continue

        # Add user message to conversation history
        conversation_history.append({"role": "user", "content": user_input})

        try:
            print("\n🤖 Mistral: ", end="", flush=True)

            # Make the API call with conversation history
            response = client.beta.conversations.start_stream(
                inputs=conversation_history,
                model="magistral-medium-latest",
                instructions="You are a helpful AI assistant. Be conversational and friendly.",
                completion_args=completion_args,
                tools=tools,
            )

            # Process and print the stream
            assistant_response = print_response(response)

            # Add assistant response to conversation history
            if assistant_response.strip():
                conversation_history.append(
                    {"role": "assistant", "content": assistant_response.strip()}
                )

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'quit' to exit.")


if __name__ == "__main__":
    main()
