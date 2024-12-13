from google import genai
from google.genai import types
import base64
import os
from google.auth.exceptions import DefaultCredentialsError
from dataclasses import dataclass
from typing import List, Optional
import readline
from datetime import datetime
import pytz
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
)
import vertexai


@dataclass
class Message:
    role: str
    content: str


class ChatBot:
    def __init__(self):
        self.project_id = os.getenv("MY_GOOGLE_PROJECTID")
        self.model = None
        self.history: List[Message] = []
        self.context: List[Message] = []
        self.max_context = 10
        self.setup_readline()
        self.setup_functions()

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
            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location="us-central1")
            # Initialize Gemini model
            self.model = GenerativeModel("gemini-2.0-flash-exp")
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

    def setup_functions(self) -> None:
        """Setup function declarations for the model."""
        self.get_time_func = FunctionDeclaration(
            name="get_current_time",
            description="Get the current time in UTC and local timezone",
            parameters={
                "type": "object",
                "properties": {},
            },
        )

        self.tools = Tool(function_declarations=[self.get_time_func])

    def get_current_time(self) -> str:
        """Get current time in UTC and local timezone."""
        utc_time = datetime.now(pytz.UTC)
        local_time = datetime.now()
        local_tz = datetime.now().astimezone().tzinfo

        return (
            f"UTC Time: {utc_time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            f"Local Time ({local_tz}): {local_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    def create_contents(self) -> List[Content]:
        contents = []
        for message in self.context:
            contents.append(
                Content(role=message.role, parts=[Part.from_text(message.content)])
            )
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

            # Create the request with tools
            response = self.model.generate_content(
                contents=self.create_contents(),
                generation_config=GenerationConfig(
                    temperature=1.0,
                    top_p=0.95,
                    max_output_tokens=8192,
                ),
                tools=[self.tools],
            )

            # Handle function calls in the response
            if hasattr(response.candidates[0], "function_calls"):
                for function_call in response.candidates[0].function_calls:
                    if function_call.name == "get_current_time":
                        time_info = self.get_current_time()
                        print(time_info)
                        response_text = time_info
                        # Return early after function call to avoid error message
                        assistant_message = Message(
                            role="assistant", content=response_text
                        )
                        self.history.append(assistant_message)
                        self.update_context(assistant_message)
                        return

            # Handle regular text response
            if hasattr(response, "text") and response.text:
                print(response.text)
                response_text = response.text

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
