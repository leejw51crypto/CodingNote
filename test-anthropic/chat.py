import os
import anthropic

# Get the API key from the environment variable
api_key = os.environ["ANTHROPIC_API_KEY"]


client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=api_key,
)
message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    temperature=0.2,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ]
)
print(message.content)
