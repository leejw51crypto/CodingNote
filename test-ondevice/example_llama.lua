#!/usr/bin/env luajit
-- example_llama.lua - Example usage of llama.lua bindings

local llama = require("llama")

-- Configuration
local MODEL_PATH = os.getenv("MODEL_PATH") or "models/gemma-3-1b-q4.gguf"

-- Main function
local function main()
	print("Initializing llama.cpp backend...")
	llama.init()

	-- Load model
	print("Loading model: " .. MODEL_PATH)
	local model, err = llama.load_model(MODEL_PATH, {
		n_gpu_layers = 99, -- Offload all layers to GPU (Metal on macOS, CUDA on Linux/Windows)
		use_mmap = true,
	})

	if not model then
		print("Error: " .. err)
		llama.cleanup()
		return
	end

	print("Model loaded successfully!")
	print("Vocabulary size: " .. model:vocab_size())

	-- Create context
	local ctx, ctx_err = llama.create_context(model, {
		n_ctx = 2048,
		n_batch = 512,
		n_threads = 4,
	})

	if not ctx then
		print("Error: " .. ctx_err)
		model:free()
		llama.cleanup()
		return
	end

	print("Context created with " .. ctx:n_ctx() .. " tokens capacity")

	-- Create sampler
	local sampler = llama.create_sampler({
		temperature = 0.7,
		top_k = 40,
		top_p = 0.9,
		min_p = 0.05,
		repeat_penalty = 1.1,
		seed = 42,
	})

	-- Generate text
	local prompt = "The quick brown fox"
	print("\nPrompt: " .. prompt)
	print("Generating...")
	io.write("Response: ")

	local output, tokens = llama.generate(model, ctx, sampler, prompt, 64, function(piece)
		io.write(piece)
		io.flush()
	end)

	print("\n")
	print("Generated " .. #tokens .. " tokens")

	-- Cleanup
	sampler:free()
	ctx:free()
	model:free()
	llama.cleanup()

	print("Done!")
end

-- Run
main()
