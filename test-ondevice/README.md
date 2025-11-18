# LuaJIT llama.cpp Bindings

LuaJIT FFI bindings for running LLMs locally using llama.cpp.

## Features

- Load GGUF models
- GPU acceleration (Metal on macOS, CUDA on Linux/Windows)
- Tokenization and detokenization
- Configurable sampling (temperature, top-k, top-p, penalties)
- Streaming text generation

## Requirements

- LuaJIT
- CMake
- C++ compiler
- [Love2D](https://love2d.org/) (optional, for GUI)

## Setup

### 1. Clone with submodule

```bash
git clone --recursive <repo-url>
cd test-ondevice
```

Or if already cloned:

```bash
git submodule update --init --recursive
```

### 2. Build llama.cpp

```bash
./prepare.sh
```

### 3. Download models

```bash
./download.sh
```

This downloads:
- Gemma 2 9B Q4 (~5.4 GB)
- Llama 3.1 8B Q4 (~4.7 GB)
- Llama 3.2 3B Q4 (~1.9 GB)
- TinyLlama 1.1B Q4 (~0.6 GB)

## Usage

### Run example

```bash
./go.sh
```

### Run interactive chat

```bash
./chat.sh
```

Commands: `/model` `/settings` `/set <key> <value>` `/clear` `/quit`

### Run GUI (Love2D)

```bash
./love.sh
```

Or directly:

```bash
export LLAMA_LIB_PATH="$(pwd)/llama.cpp/build/bin/libllama.dylib"
love .
```

Requires [Love2D](https://love2d.org/) to be installed.

### Custom usage

```lua
local llama = require("llama")

-- Initialize
llama.init()

-- Load model with GPU acceleration
local model = llama.load_model("models/gemma-2-9b-q4.gguf", {
    n_gpu_layers = 99,  -- offload all layers to GPU
    use_mmap = true,
})

-- Create context
local ctx = llama.create_context(model, {
    n_ctx = 2048,
    n_batch = 512,
    n_threads = 4,
})

-- Create sampler
local sampler = llama.create_sampler({
    temperature = 0.7,
    top_k = 40,
    top_p = 0.9,
    min_p = 0.05,
    repeat_penalty = 1.1,
    seed = 42,
})

-- Generate text with streaming
local output = llama.generate(model, ctx, sampler, "Hello", 128, function(piece)
    io.write(piece)
    io.flush()
end)

-- Cleanup
sampler:free()
ctx:free()
model:free()
llama.cleanup()
```

## API Reference

### Module Functions

#### `llama.init()`
Initialize the llama.cpp backend. Call once at program start.

#### `llama.cleanup()`
Free backend resources. Call once at program end.

#### `llama.load_model(path, params)`
Load a GGUF model file.

Parameters:
- `path`: Path to the model file
- `params.n_gpu_layers`: Number of layers to offload to GPU (99 = all)
- `params.use_mmap`: Use memory-mapped file (default: true)
- `params.use_mlock`: Lock model in RAM

Returns: Model object or nil, error message

#### `llama.create_context(model, params)`
Create an inference context.

Parameters:
- `model`: Model object
- `params.n_ctx`: Context size (default: from model)
- `params.n_batch`: Batch size (default: 512)
- `params.n_threads`: Number of threads
- `params.embeddings`: Extract embeddings (default: false)

Returns: Context object or nil, error message

#### `llama.create_sampler(params)`
Create a token sampler.

Parameters:
- `params.temperature`: Sampling temperature (0 = greedy)
- `params.top_k`: Top-k sampling
- `params.top_p`: Top-p (nucleus) sampling
- `params.min_p`: Min-p sampling
- `params.repeat_penalty`: Repetition penalty
- `params.freq_penalty`: Frequency penalty
- `params.presence_penalty`: Presence penalty
- `params.penalty_last_n`: Tokens to consider for penalties
- `params.seed`: Random seed

Returns: Sampler object

#### `llama.generate(model, ctx, sampler, prompt, max_tokens, callback)`
Generate text from a prompt.

Parameters:
- `model`: Model object
- `ctx`: Context object
- `sampler`: Sampler object
- `prompt`: Input text
- `max_tokens`: Maximum tokens to generate
- `callback`: Function called for each token: `callback(piece, token)`

Returns: Generated text, token array

### Model Methods

- `model:free()` - Free model resources
- `model:vocab_size()` - Get vocabulary size
- `model:bos_token()` - Get BOS token ID
- `model:eos_token()` - Get EOS token ID
- `model:is_eog(token)` - Check if token is end-of-generation
- `model:tokenize(text)` - Convert text to tokens
- `model:token_to_piece(token)` - Convert token to text
- `model:detokenize(tokens)` - Convert tokens to text

### Context Methods

- `ctx:free()` - Free context resources
- `ctx:n_ctx()` - Get context size
- `ctx:decode(tokens)` - Process tokens

### Sampler Methods

- `sampler:free()` - Free sampler resources
- `sampler:sample(ctx, idx)` - Sample next token
- `sampler:accept(token)` - Accept token

## Environment Variables

- `LLAMA_LIB_PATH`: Path to libllama shared library
- `MODEL_PATH`: Path to model file

## Files

```
test-ondevice/
├── llama.lua          # LuaJIT FFI bindings
├── ai.lua             # Interactive chat interface
├── main.lua           # Love2D GUI application
├── example_llama.lua  # Usage example
├── go.sh              # Run example
├── chat.sh            # Run interactive chat
├── love.sh            # Run Love2D GUI
├── prepare.sh         # Build llama.cpp
├── download.sh        # Download models
├── llama.cpp/         # llama.cpp submodule
└── models/            # Model files (gitignored)
```

## License

MIT
