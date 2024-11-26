import asyncio
import json
import os
import pyaudio
import websockets
from dotenv import load_dotenv
from openai import OpenAI
import sounddevice as sd
import soundfile as sf
from deepgram import DeepgramClient, SpeakOptions

load_dotenv()

USE_VOICE = False  # Global voice flag

class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

class LanguageModelProcessor:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.messages = [
            # Load the system prompt from file
            {"role": "system", "content": open('system_prompt.txt', 'r').read().strip()}
        ]

    def process(self, text):
        self.messages.append({"role": "user", "content": text})
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=self.messages,
            temperature=0
        )
        
        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message

async def generate_and_play_speech(text, output_file="output.wav"):
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise ValueError("Please set DEEPGRAM_API_KEY environment variable")

    deepgram = DeepgramClient(api_key)

    try:
        options = SpeakOptions(
            model="aura-asteria-en",
        )

        await deepgram.speak.asyncrest.v("1").save(
            output_file,
            {"text": text},
            options
        )

        data, samplerate = sf.read(output_file)
        sd.play(data, samplerate)
        sd.wait()

    except Exception as e:
        print(f"Error generating or playing speech: {str(e)}")

async def get_transcript(callback):
    transcription_complete = asyncio.Event()
    transcript_collector = TranscriptCollector()

    try:
        DEEPGRAM_URL = f"wss://api.deepgram.com/v1/listen?model=nova-2&punctuate=true&language=en-US&encoding=linear16&channels=1&sample_rate=16000&endpointing=300&smart_format=true"
        
        async with websockets.connect(
            DEEPGRAM_URL,
            extra_headers={"Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY')}"}
        ) as ws:
            print("Listening...")

            async def send_audio():
                CHUNK = 1024
                FORMAT = pyaudio.paInt16
                CHANNELS = 1
                RATE = 16000

                p = pyaudio.PyAudio()
                stream = p.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=None,
                    stream_callback=None
                )
                
                while True:
                    try:
                        data = stream.read(CHUNK, exception_on_overflow=False)
                        await ws.send(data)
                        await asyncio.sleep(0.01)
                    except IOError as e:
                        if e.errno == pyaudio.paInputOverflowed:
                            print("Input overflow, dropping data")
                            continue
                        print(f"IOError in send_audio: {e}")
                    except Exception as e:
                        print(f"Error in send_audio: {e}")
                        break
                    
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

            send_task = asyncio.create_task(send_audio())
            receive_task = asyncio.create_task(receive_transcription())
            
            await transcription_complete.wait()
            
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

    async def get_text_input(self):
        text = input("You: ")
        return text

    async def main(self):
        global USE_VOICE  # Reference the global variable
        
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        while True:
            if USE_VOICE:
                await get_transcript(handle_full_sentence)
                user_input = self.transcription_response
            else:
                user_input = await self.get_text_input()
            
            if "goodbye" in user_input.lower():
                print("Goodbye!")
                break
            
            llm_response = self.llm.process(user_input)
            print(f"Assistant: {llm_response}")
            
            if USE_VOICE:
                await generate_and_play_speech(llm_response)
            self.transcription_response = ""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-voice", action="store_true", help="Disable voice mode")
    args = parser.parse_args()
    
    if args.no_voice:
        USE_VOICE = False

    manager = ConversationManager()
    asyncio.run(manager.main())
