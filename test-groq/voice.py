# source from: https://github.com/gkamradt/QuickAgent
import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os
import websockets
import json
import pyaudio

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

load_dotenv()

class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview", openai_api_key=os.getenv("OPENAI_API_KEY"))
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125", openai_api_key=os.getenv("OPENAI_API_KEY"))

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Load the system prompt from a file
        with open('system_prompt.txt', 'r') as file:
            system_prompt = file.read().strip()
        
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )

    def process(self, text):
        self.memory.chat_memory.add_user_message(text)  # Add user message to memory

        start_time = time.time()

        # Go get the response from the LLM
        response = self.conversation.invoke({"text": text})
        end_time = time.time()

        self.memory.chat_memory.add_ai_message(response['text'])  # Add AI response to memory

        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM ({elapsed_time}ms): {response['text']}")
        return response['text']

class TextToSpeech:
    # Set your Deepgram API Key and desired voice model
    DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    MODEL_NAME = "aura-helios-en"  # Example model name, change as needed

    @staticmethod
    def is_installed(lib_name: str) -> bool:
        lib = shutil.which(lib_name)
        return lib is not None

    def speak(self, text):
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found, necessary to stream audio.")

        DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}&performance=some&encoding=linear16&sample_rate=24000"
        headers = {
            "Authorization": f"Token {self.DG_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text
        }

        player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
        player_process = subprocess.Popen(
            player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        start_time = time.time()  # Record the time before sending the request
        first_byte_time = None  # Initialize a variable to store the time when the first byte is received

        with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    if first_byte_time is None:  # Check if this is the first chunk received
                        first_byte_time = time.time()  # Record the time when the first byte is received
                        ttfb = int((first_byte_time - start_time)*1000)  # Calculate the time to first byte
                        print(f"TTS Time to First Byte (TTFB): {ttfb}ms\n")
                    player_process.stdin.write(chunk)
                    player_process.stdin.flush()

        if player_process.stdin:
            player_process.stdin.close()
        player_process.wait()

class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

transcript_collector = TranscriptCollector()

async def get_transcript(callback):
    transcription_complete = asyncio.Event()
    transcript_collector = TranscriptCollector()

    try:
        # Initialize Deepgram API URL and authentication
        DEEPGRAM_URL = f"wss://api.deepgram.com/v1/listen?model=nova-2&punctuate=true&language=en-US&encoding=linear16&channels=1&sample_rate=16000&endpointing=300&smart_format=true"
        
        async with websockets.connect(
            DEEPGRAM_URL,
            extra_headers={"Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY')}"}
        ) as ws:
            print("Listening...")

            # Open microphone stream
            audio_queue = asyncio.Queue()
            
            async def send_audio():
                p = pyaudio.PyAudio()
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=8000
                )
                
                try:
                    while True:
                        data = stream.read(8000)
                        await ws.send(data)
                        await asyncio.sleep(0.1)
                except Exception as e:
                    print(f"Error in send_audio: {e}")
                finally:
                    stream.stop_stream()
                    stream.close()
                    p.terminate()

            async def receive_transcription():
                try:
                    while True:
                        msg = await ws.recv()
                        result = json.loads(msg)
                        
                        if "channel" in result:
                            sentence = result["channel"]["alternatives"][0]["transcript"]
                            
                            if not result.get("is_final"):
                                transcript_collector.add_part(sentence)
                            else:
                                transcript_collector.add_part(sentence)
                                full_sentence = transcript_collector.get_full_transcript()
                                if len(full_sentence.strip()) > 0:
                                    full_sentence = full_sentence.strip()
                                    print(f"Human: {full_sentence}")
                                    callback(full_sentence)
                                    transcript_collector.reset()
                                    transcription_complete.set()
                                    return
                except Exception as e:
                    print(f"Error in receive_transcription: {e}")

            # Create tasks for sending audio and receiving transcriptions
            send_task = asyncio.create_task(send_audio())
            receive_task = asyncio.create_task(receive_transcription())
            
            # Wait for transcription to complete
            await transcription_complete.wait()
            
            # Cancel the tasks
            send_task.cancel()
            receive_task.cancel()
            
            try:
                await send_task
                await receive_task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        print(f"Could not open websocket connection: {e}")
        return

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        # Loop indefinitely until "goodbye" is detected
        while True:
            await get_transcript(handle_full_sentence)
            
            # Check for "goodbye" to exit the loop
            if "goodbye" in self.transcription_response.lower():
                break
            
            llm_response = self.llm.process(self.transcription_response)

            tts = TextToSpeech()
            tts.speak(llm_response)

            # Reset transcription_response for the next loop iteration
            self.transcription_response = ""

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())
