#!/usr/bin/env luajit
-- ai.lua - High-level interactive AI chat interface

local ffi = require("ffi")
local llama = require("llama")

-- JSON library (minimal implementation)
local json = {}

function json.encode(obj)
	if type(obj) == "table" then
		if #obj > 0 then
			-- Array
			local items = {}
			for _, v in ipairs(obj) do
				table.insert(items, json.encode(v))
			end
			return "[" .. table.concat(items, ",") .. "]"
		else
			-- Object
			local items = {}
			for k, v in pairs(obj) do
				table.insert(items, string.format('"%s":%s', k, json.encode(v)))
			end
			return "{" .. table.concat(items, ",") .. "}"
		end
	elseif type(obj) == "string" then
		return string.format('"%s"', obj:gsub("\\", "\\\\"):gsub('"', '\\"'):gsub("\n", "\\n"))
	elseif type(obj) == "number" or type(obj) == "boolean" then
		return tostring(obj)
	elseif obj == nil then
		return "null"
	end
	return '""'
end

function json.decode(str)
	-- Simple JSON parser
	local pos = 1

	local function skip_whitespace()
		while pos <= #str and str:sub(pos, pos):match("%s") do
			pos = pos + 1
		end
	end

	local function parse_string()
		pos = pos + 1 -- skip opening quote
		local start = pos
		local result = ""
		while pos <= #str do
			local c = str:sub(pos, pos)
			if c == '"' then
				pos = pos + 1
				return result
			elseif c == "\\" then
				pos = pos + 1
				local next_c = str:sub(pos, pos)
				if next_c == "n" then
					result = result .. "\n"
				elseif next_c == "t" then
					result = result .. "\t"
				elseif next_c == "\\" then
					result = result .. "\\"
				elseif next_c == '"' then
					result = result .. '"'
				else
					result = result .. next_c
				end
			else
				result = result .. c
			end
			pos = pos + 1
		end
		return result
	end

	local function parse_number()
		local start = pos
		while pos <= #str and str:sub(pos, pos):match("[%d%.%-eE%+]") do
			pos = pos + 1
		end
		return tonumber(str:sub(start, pos - 1))
	end

	local parse_value

	local function parse_array()
		local arr = {}
		pos = pos + 1 -- skip [
		skip_whitespace()
		if str:sub(pos, pos) == "]" then
			pos = pos + 1
			return arr
		end
		while true do
			table.insert(arr, parse_value())
			skip_whitespace()
			if str:sub(pos, pos) == "]" then
				pos = pos + 1
				return arr
			end
			pos = pos + 1 -- skip comma
			skip_whitespace()
		end
	end

	local function parse_object()
		local obj = {}
		pos = pos + 1 -- skip {
		skip_whitespace()
		if str:sub(pos, pos) == "}" then
			pos = pos + 1
			return obj
		end
		while true do
			skip_whitespace()
			local key = parse_string()
			skip_whitespace()
			pos = pos + 1 -- skip :
			skip_whitespace()
			obj[key] = parse_value()
			skip_whitespace()
			if str:sub(pos, pos) == "}" then
				pos = pos + 1
				return obj
			end
			pos = pos + 1 -- skip comma
		end
	end

	parse_value = function()
		skip_whitespace()
		local c = str:sub(pos, pos)
		if c == '"' then
			return parse_string()
		elseif c == "[" then
			return parse_array()
		elseif c == "{" then
			return parse_object()
		elseif c == "t" then
			pos = pos + 4
			return true
		elseif c == "f" then
			pos = pos + 5
			return false
		elseif c == "n" then
			pos = pos + 4
			return nil
		else
			return parse_number()
		end
	end

	return parse_value()
end

-- Configuration
local CONFIG_FILE = "info.json"
local MODELS_DIR = "models"
local DEBUG = os.getenv("DEBUG") == "1"

-- Suppress llama.cpp debug output if not in debug mode
if not DEBUG then
	-- Redirect stderr to /dev/null using C library
	ffi.cdef([[
		int dup2(int oldfd, int newfd);
		int open(const char *pathname, int flags);
		int close(int fd);
	]])
	local devnull = ffi.C.open("/dev/null", 1) -- O_WRONLY = 1
	if devnull >= 0 then
		ffi.C.dup2(devnull, 2) -- redirect stderr (fd 2)
		ffi.C.close(devnull)
	end
end

-- Default configuration
local default_config = {
	model = "",
	n_ctx = 4096,
	n_batch = 512,
	n_threads = 4,
	n_gpu_layers = 99,
	temperature = 0.7,
	top_k = 40,
	top_p = 0.9,
	min_p = 0.05,
	repeat_penalty = 1.1,
	max_tokens = 512,
	system_prompt = "You are a helpful AI assistant.",
}

-- AI class
local AI = {}
AI.__index = AI

function AI.new()
	local self = setmetatable({}, AI)
	self.config = nil
	self.model = nil
	self.ctx = nil
	self.sampler = nil
	self.history = {}
	return self
end

function AI:load_config()
	local f = io.open(CONFIG_FILE, "r")
	if f then
		local content = f:read("*all")
		f:close()
		local ok, cfg = pcall(json.decode, content)
		if ok and cfg then
			-- Merge with defaults
			for k, v in pairs(default_config) do
				if cfg[k] == nil then
					cfg[k] = v
				end
			end
			self.config = cfg
			return true
		end
	end
	self.config = {}
	for k, v in pairs(default_config) do
		self.config[k] = v
	end
	return false
end

function AI:save_config()
	local f = io.open(CONFIG_FILE, "w")
	if f then
		f:write(json.encode(self.config))
		f:close()
		return true
	end
	return false
end

function AI:list_models()
	local models = {}
	local p = io.popen('ls -1 "' .. MODELS_DIR .. '"/*.gguf 2>/dev/null')
	if p then
		for line in p:lines() do
			local name = line:match("([^/]+)$")
			if name then
				table.insert(models, name)
			end
		end
		p:close()
	end
	return models
end

function AI:select_model()
	local models = self:list_models()
	if #models == 0 then
		print("No models found in " .. MODELS_DIR .. "/")
		print("Run ./download.sh to download models")
		return false
	end

	print("\nAvailable models:")
	for i, name in ipairs(models) do
		local marker = ""
		if self.config.model == name then
			marker = " [current]"
		end
		print(string.format("  %d. %s%s", i, name, marker))
	end

	io.write("\nSelect model (1-" .. #models .. "): ")
	local choice = io.read()
	local idx = tonumber(choice)

	if idx and idx >= 1 and idx <= #models then
		self.config.model = models[idx]
		self:save_config()
		print("Selected: " .. self.config.model)
		return true
	else
		print("Invalid selection")
		return false
	end
end

function AI:init()
	if not self.config.model or self.config.model == "" then
		print("No model selected.")
		if not self:select_model() then
			return false
		end
	end

	local model_path = MODELS_DIR .. "/" .. self.config.model

	-- Check if model exists
	local f = io.open(model_path, "r")
	if not f then
		print("Model not found: " .. model_path)
		if not self:select_model() then
			return false
		end
		model_path = MODELS_DIR .. "/" .. self.config.model
	else
		f:close()
	end

	print("Loading model: " .. self.config.model)

	-- Initialize backend
	llama.init()

	-- Load model
	local model, err = llama.load_model(model_path, {
		n_gpu_layers = self.config.n_gpu_layers,
		use_mmap = true,
	})

	if not model then
		print("Error loading model: " .. (err or "unknown error"))
		llama.cleanup()
		return false
	end

	self.model = model

	-- Create context
	local ctx, ctx_err = llama.create_context(model, {
		n_ctx = self.config.n_ctx,
		n_batch = self.config.n_batch,
		n_threads = self.config.n_threads,
	})

	if not ctx then
		print("Error creating context: " .. (ctx_err or "unknown error"))
		model:free()
		llama.cleanup()
		return false
	end

	self.ctx = ctx

	-- Create sampler
	self.sampler = llama.create_sampler({
		temperature = self.config.temperature,
		top_k = self.config.top_k,
		top_p = self.config.top_p,
		min_p = self.config.min_p,
		repeat_penalty = self.config.repeat_penalty,
		seed = os.time(),
	})

	print("Model loaded successfully!")
	print("Context size: " .. ctx:n_ctx() .. " tokens")

	-- Show acceleration info
	if self.config.n_gpu_layers > 0 then
		local platform = ffi.os
		if platform == "OSX" then
			print("Acceleration: Metal GPU (" .. self.config.n_gpu_layers .. " layers)")
		elseif platform == "Linux" or platform == "Windows" then
			print("Acceleration: CUDA GPU (" .. self.config.n_gpu_layers .. " layers)")
		else
			print("Acceleration: GPU (" .. self.config.n_gpu_layers .. " layers)")
		end
	else
		print("Acceleration: CPU only")
	end
	print("")

	return true
end

function AI:cleanup()
	if self.sampler then
		self.sampler:free()
		self.sampler = nil
	end
	if self.ctx then
		self.ctx:free()
		self.ctx = nil
	end
	if self.model then
		self.model:free()
		self.model = nil
	end
	llama.cleanup()
end

function AI:format_prompt(user_input)
	-- Build conversation with system prompt
	local prompt = self.config.system_prompt .. "\n\n"

	-- Add history
	for _, msg in ipairs(self.history) do
		if msg.role == "user" then
			prompt = prompt .. "User: " .. msg.content .. "\n"
		else
			prompt = prompt .. "Assistant: " .. msg.content .. "\n"
		end
	end

	-- Add current input
	prompt = prompt .. "User: " .. user_input .. "\nAssistant:"

	return prompt
end

function AI:chat(user_input)
	local prompt = self:format_prompt(user_input)

	-- Generate response
	local response = ""
	local output, tokens = llama.generate(
		self.model,
		self.ctx,
		self.sampler,
		prompt,
		self.config.max_tokens,
		function(piece)
			io.write(piece)
			io.flush()
			response = response .. piece
		end
	)

	print("") -- newline after response

	-- Add to history
	table.insert(self.history, { role = "user", content = user_input })
	table.insert(self.history, { role = "assistant", content = response })

	-- Trim history if too long (keep last 10 exchanges)
	while #self.history > 20 do
		table.remove(self.history, 1)
	end

	return response
end

function AI:clear_history()
	self.history = {}
	print("Conversation history cleared.")
end

function AI:show_settings()
	print("\nCurrent settings:")
	print("  model: " .. (self.config.model or "not set"))
	print("  n_ctx: " .. self.config.n_ctx)
	print("  n_gpu_layers: " .. self.config.n_gpu_layers)
	print("  temperature: " .. self.config.temperature)
	print("  top_k: " .. self.config.top_k)
	print("  top_p: " .. self.config.top_p)
	print("  max_tokens: " .. self.config.max_tokens)
	print("  system_prompt: " .. self.config.system_prompt:sub(1, 50) .. "...")
	print("")
end

function AI:set_setting(key, value)
	if self.config[key] ~= nil then
		if type(self.config[key]) == "number" then
			value = tonumber(value)
		end
		self.config[key] = value
		self:save_config()
		print("Set " .. key .. " = " .. tostring(value))
	else
		print("Unknown setting: " .. key)
	end
end

-- Main interactive loop
local function main()
	local ai = AI.new()

	-- Load config
	ai:load_config()

	print("=== AI Chat ===")
	print("Commands: /model /settings /set <key> <value> /clear /quit")
	print("")

	-- Initialize
	if not ai:init() then
		return
	end

	-- Interactive loop
	while true do
		io.write("\nYou: ")
		local input = io.read()

		if not input then
			break
		end

		input = input:match("^%s*(.-)%s*$") -- trim

		if input == "" then
			-- skip empty input
		elseif input == "/quit" or input == "/exit" or input == "/q" then
			break
		elseif input == "/model" then
			ai:cleanup()
			if ai:select_model() then
				ai:init()
			end
		elseif input == "/settings" then
			ai:show_settings()
		elseif input:match("^/set%s+") then
			local key, value = input:match("^/set%s+(%S+)%s+(.+)$")
			if key and value then
				ai:set_setting(key, value)
			else
				print("Usage: /set <key> <value>")
			end
		elseif input == "/clear" then
			ai:clear_history()
		elseif input:sub(1, 1) == "/" then
			print("Unknown command. Commands: /model /settings /set /clear /quit")
		else
			io.write("\nAssistant: ")
			ai:chat(input)
		end
	end

	-- Cleanup
	print("\nGoodbye!")
	ai:cleanup()
end

-- Run
main()
