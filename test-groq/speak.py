import os
import asyncio
import sounddevice as sd
import soundfile as sf
from deepgram import DeepgramClient, SpeakOptions

async def generate_and_play_speech(text, output_file="output.wav"):
    # Get API key from environment variable
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise ValueError("Please set DEEPGRAM_API_KEY environment variable")

    # Initialize the Deepgram client
    deepgram = DeepgramClient(api_key)

    try:
        # Configure TTS options
        options = SpeakOptions(
            model="aura-asteria-en",
        )

        # Generate audio from text
        response = await deepgram.speak.asyncrest.v("1").save(
            output_file,
            {"text": text},
            options
        )
        
        print(f"Audio saved to {output_file}")

        # Play the generated audio
        data, samplerate = sf.read(output_file)
        sd.play(data, samplerate)
        sd.wait()  # Wait until the audio is finished playing

    except Exception as e:
        print(f"Error generating or playing speech: {str(e)}")

# Example usage
if __name__ == "__main__":
    # First, make sure you have set your API key in the environment:
    # export DEEPGRAM_API_KEY='your-api-key'
    
    text = "buy bitcoin"
    asyncio.run(generate_and_play_speech(text))
