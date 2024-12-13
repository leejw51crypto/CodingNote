from google import genai
from google.genai import types
import base64
import os
from google.auth.exceptions import DefaultCredentialsError
from dataclasses import dataclass
from typing import List, Optional
import readline  # Add this at the top with other imports


@dataclass
class Message:
    role: str
    content: str


class ChatBot:
    def __init__(self):
        self.project_id = os.getenv("MY_GOOGLE_PROJECTID")
        self.client = None
        self.history: List[Message] = []  # Full history
        self.context: List[Message] = []  # Active context window
        self.max_context = 10  # Maximum number of message pairs to keep in context
        self.setup_readline()

    def setup_readline(self) -> None:
        # Enable history file
        histfile = os.path.join(os.path.expanduser("~"), ".chat_history")
        try:
            readline.read_history_file(histfile)
            # Set maximum number of items that will be written to the history file
            readline.set_history_length(1000)
        except FileNotFoundError:
            pass

        # Write history on exit
        import atexit

        atexit.register(readline.write_history_file, histfile)

    def initialize(self) -> bool:
        if not self.project_id:
            print("Error: Environment variable MY_GOOGLE_PROJECTID is not set")
            return False

        print(f"Successfully loaded project ID: {self.project_id}")

        try:
            self.client = genai.Client(
                vertexai=True, project=self.project_id, location="us-central1"
            )
            return True
        except DefaultCredentialsError:
            print("\nError: Google Cloud credentials not found!")
            print("Please set up your credentials using one of these methods:")
            print("1. Run: gcloud auth application-default login")
            print(
                "2. Set GOOGLE_APPLICATION_CREDENTIALS environment variable pointing to your service account key file"
            )
            print(
                "\nFor more information, visit: https://cloud.google.com/docs/authentication/external/set-up-adc"
            )
            return False

    def get_generate_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=8192,
            response_modalities=["TEXT"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT", threshold="OFF"
                ),
            ],
        )

    def create_contents(self) -> List[types.Content]:
        # Use context instead of history for generating responses
        contents = []
        for message in self.context:
            contents.append(
                types.Content(
                    role=message.role, parts=[types.Part(text=message.content)]
                )
            )
        # debug print
        # print(f"Contents: {contents}")
        return contents

    def update_context(self, message: Message) -> None:
        """Update the context window while maintaining size limit."""
        self.context.append(message)
        # Keep only the most recent message pairs within max_context limit
        if len(self.context) > self.max_context * 2:
            # Remove oldest message pair
            self.context = self.context[2:]

    def generate_response(self, user_input: str) -> None:
        user_message = Message(role="user", content=user_input)
        self.history.append(user_message)
        self.update_context(user_message)

        try:
            print("Gemini: ", end="")
            response_text = ""
            for chunk in self.client.models.generate_content_stream(
                model="gemini-2.0-flash-exp",
                contents=self.create_contents(),
                config=self.get_generate_config(),
            ):
                if chunk.candidates:
                    chunk_text = chunk.candidates[0].content.parts[0].text
                    response_text += chunk_text
                    print(chunk_text, end="")
            print("\n")

            assistant_message = Message(role="assistant", content=response_text)
            self.history.append(assistant_message)
            self.update_context(assistant_message)

        except Exception as e:
            if "429" in str(e):
                print(
                    "Error: Rate limit exceeded. Please wait a moment before trying again."
                )
                print(
                    "For more information, visit: https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429"
                )
            else:
                print(f"Error occurred: {str(e)}")
            print("\n")

    def chat_loop(self) -> None:
        print("\nChat started! (Type 'quit' to exit, 'context' to see current context)")

        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() == "quit":
                    print("Goodbye!")
                    break

                if user_input.lower() == "context":
                    print("\nCurrent context window:")
                    for msg in self.context:
                        print(f"{msg.role}: {msg.content}")
                    continue

                if not user_input:
                    continue

                self.generate_response(user_input)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break


def main():
    chatbot = ChatBot()
    if chatbot.initialize():
        chatbot.chat_loop()


if __name__ == "__main__":
    main()
