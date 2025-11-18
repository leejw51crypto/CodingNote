#!/bin/bash

# Download GGUF models from HuggingFace

mkdir -p models

echo "Downloading Gemma 2 9B Q4..."
curl -L -o models/gemma-2-9b-q4.gguf \
    "https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf"

echo "Downloading Llama 3.1 8B Q4..."
curl -L -o models/llama-3.1-8b-q4.gguf \
    "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

echo "Downloading Llama 3.2 3B Q4..."
curl -L -o models/llama-3.2-3b-q4.gguf \
    "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

echo "Downloading TinyLlama 1.1B Q4..."
curl -L -o models/tinyllama-1.1b-q4.gguf \
    "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

echo "Done!"
ls -lh models/
