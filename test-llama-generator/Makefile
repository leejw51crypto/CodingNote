all: mistral 

build:
	cargo build

cuda:
	cargo run --features cublas --release -- --n-len 1024 "what is rust lang?" hf-model TheBloke/Mistral-7B-Instruct-v0.2-GGUF  mistral-7b-instruct-v0.2.Q2_K.gguf

cuda2:
	./target/release/test-llama-generator  local ~/models/ggml-model-q4_0.gguf

mistral:
	cargo run --release -- --n-len 1024 "what is rust lang?" hf-model TheBloke/Mistral-7B-Instruct-v0.2-GGUF  mistral-7b-instruct-v0.2.Q2_K.gguf

mixtral:
	cargo run --release -- --n-len 1024 "what is rust lang?" hf-model TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF  mixtral-8x7b-instruct-v0.1.Q4_0.gguf

gemma:
	cargo run --release -- --n-len 1024 "what is rust lang?" hf-model lmstudio-ai/gemma-2b-it-GGUF  gemma-2b-it-q4_k_m.gguf

llama:
	cargo run --release -- --n-len 1024 "what is rust lang?" hf-model TheBloke/Llama-2-13B-chat-GGUF   llama-2-13b-chat.Q4_K_M.gguf
