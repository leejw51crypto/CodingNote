all: build

build:
	cargo build 

mistral:
	cargo run --release -- --n-len 1024 "what is rust lang?" hf-model TheBloke/Mistral-7B-Instruct-v0.2-GGUF  mistral-7b-instruct-v0.2.Q2_K.gguf
