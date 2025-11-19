-- main.lua - Love2D GUI for AI Chat
-- Run with: love .

local ffi = require("ffi")

-- Suppress llama.cpp debug output
ffi.cdef([[
    int dup2(int oldfd, int newfd);
    int open(const char *pathname, int flags);
    int close(int fd);
]])
local devnull = ffi.C.open("/dev/null", 1)
if devnull >= 0 then
	ffi.C.dup2(devnull, 2)
	ffi.C.close(devnull)
end

-- Set library path for Love2D (doesn't inherit shell env)
local os_name = ffi.os
local lib_name
if os_name == "OSX" then
	lib_name = "libllama.dylib"
elseif os_name == "Windows" then
	lib_name = "llama.dll"
else
	lib_name = "libllama.so"
end

local lib_path = love.filesystem.getSource() .. "/" .. lib_name
-- Check if running from fused app or directory
if not love.filesystem.getInfo(lib_name) then
	-- Fallback paths
	lib_path = os.getenv("LLAMA_LIB_PATH") or lib_name
end

-- Override the environment variable for llama.lua
package.loaded["llama"] = nil
os.getenv = (function(original)
	return function(key)
		if key == "LLAMA_LIB_PATH" then
			return lib_path
		end
		return original(key)
	end
end)(os.getenv)

local llama = require("llama")

-- JSON library (minimal)
local json = {}

function json.encode(obj)
	if type(obj) == "table" then
		if #obj > 0 then
			local items = {}
			for _, v in ipairs(obj) do
				table.insert(items, json.encode(v))
			end
			return "[" .. table.concat(items, ",") .. "]"
		else
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
	local pos = 1
	local function skip_whitespace()
		while pos <= #str and str:sub(pos, pos):match("%s") do
			pos = pos + 1
		end
	end
	local function parse_string()
		pos = pos + 1
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
		pos = pos + 1
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
			pos = pos + 1
			skip_whitespace()
		end
	end
	local function parse_object()
		local obj = {}
		pos = pos + 1
		skip_whitespace()
		if str:sub(pos, pos) == "}" then
			pos = pos + 1
			return obj
		end
		while true do
			skip_whitespace()
			local key = parse_string()
			skip_whitespace()
			pos = pos + 1
			skip_whitespace()
			obj[key] = parse_value()
			skip_whitespace()
			if str:sub(pos, pos) == "}" then
				pos = pos + 1
				return obj
			end
			pos = pos + 1
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

-- App state
local app = {
	-- UI
	width = 800,
	height = 600,
	font = nil,
	small_font = nil,

	-- Input
	input_text = "",
	cursor_blink = 0,

	-- Messages
	messages = {},
	scroll_y = 0,
	scroll_target = 0,
	scroll_velocity = 0,

	-- AI
	model = nil,
	ctx = nil,
	sampler = nil,
	generating = false,
	current_response = "",

	-- Config
	config = {
		model = "gemma-3-1b-q4.gguf",
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
	},

	-- State
	status = "Loading...",
	models_list = {},
	show_model_select = false,
	selected_model_idx = 1,
}

function app.load_config()
	local f = io.open("info.json", "r")
	if f then
		local content = f:read("*all")
		f:close()
		local ok, cfg = pcall(json.decode, content)
		if ok and cfg then
			for k, v in pairs(cfg) do
				if app.config[k] ~= nil then
					app.config[k] = v
				end
			end
		end
	end
end

function app.save_config()
	local f = io.open("info.json", "w")
	if f then
		f:write(json.encode(app.config))
		f:close()
	end
end

function app.list_models()
	local models = {}
	local p = io.popen("ls -1 models/*.gguf 2>/dev/null")
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

function app.init_ai()
	if app.config.model == "" then
		app.models_list = app.list_models()
		if #app.models_list > 0 then
			app.show_model_select = true
			app.status = "Select a model"
		else
			app.status = "No models found. Run ./download.sh"
		end
		return false
	end

	local model_path = "models/" .. app.config.model
	local f = io.open(model_path, "r")
	if not f then
		app.status = "Model not found: " .. app.config.model
		app.config.model = ""
		return app.init_ai()
	end
	f:close()

	app.status = "Loading " .. app.config.model .. "..."

	-- Initialize in coroutine to not block UI
	return true
end

function app.load_model_async()
	llama.init()

	local model_path = "models/" .. app.config.model
	local model, err = llama.load_model(model_path, {
		n_gpu_layers = app.config.n_gpu_layers,
		use_mmap = true,
	})

	if not model then
		app.status = "Error: " .. (err or "unknown")
		return false
	end

	app.model = model

	local ctx, ctx_err = llama.create_context(model, {
		n_ctx = app.config.n_ctx,
		n_batch = app.config.n_batch,
		n_threads = app.config.n_threads,
	})

	if not ctx then
		app.status = "Error: " .. (ctx_err or "unknown")
		model:free()
		return false
	end

	app.ctx = ctx

	app.sampler = llama.create_sampler({
		temperature = app.config.temperature,
		top_k = app.config.top_k,
		top_p = app.config.top_p,
		min_p = app.config.min_p,
		repeat_penalty = app.config.repeat_penalty,
		seed = os.time(),
	})

	local accel = app.config.n_gpu_layers > 0 and "GPU" or "CPU"
	app.status = app.config.model .. " | " .. accel .. " | " .. ctx:n_ctx() .. " ctx"

	return true
end

function app.format_prompt(user_input)
	local prompt = app.config.system_prompt .. "\n\n"
	-- Exclude the last 2 messages (current user + empty assistant placeholder)
	local history_end = #app.messages - 2
	for i = 1, history_end do
		local msg = app.messages[i]
		if msg.role == "user" then
			prompt = prompt .. "User: " .. msg.content .. "\n"
		else
			prompt = prompt .. "Assistant: " .. msg.content .. "\n"
		end
	end
	prompt = prompt .. "User: " .. user_input .. "\nAssistant:"
	return prompt
end

function app.send_message()
	if app.input_text == "" or app.generating then
		return
	end
	if not app.model then
		return
	end

	local user_msg = app.input_text
	app.input_text = ""

	table.insert(app.messages, { role = "user", content = user_msg })
	table.insert(app.messages, { role = "assistant", content = "" })

	app.generating = true
	app.current_response = ""

	-- Start generation in coroutine
	app.gen_coroutine = coroutine.create(function()
		local prompt = app.format_prompt(user_msg)
		local tokens = app.model:tokenize(prompt)

		-- Clear KV cache and reset sampler before each generation
		app.ctx:clear_kv_cache()
		app.sampler:reset()

		if not app.ctx:decode(tokens) then
			app.messages[#app.messages].content = "[Error: Failed to decode]"
			return
		end

		for _ = 1, app.config.max_tokens do
			local new_token = app.sampler:sample(app.ctx, -1)

			if app.model:is_eog(new_token) then
				break
			end

			app.sampler:accept(new_token)
			local piece = app.model:token_to_piece(new_token)
			app.current_response = app.current_response .. piece
			app.messages[#app.messages].content = app.current_response

			coroutine.yield()

			if not app.ctx:decode({ new_token }) then
				break
			end
		end
	end)
end

function app.cleanup()
	if app.sampler then
		app.sampler:free()
	end
	if app.ctx then
		app.ctx:free()
	end
	if app.model then
		app.model:free()
	end
	llama.cleanup()
end

-- Love2D callbacks

function love.load()
	love.window.setTitle("AI Chat")
	love.window.setMode(app.width, app.height, { resizable = true })

	app.font = love.graphics.newFont(24)
	app.small_font = love.graphics.newFont(18)
	love.graphics.setFont(app.font)

	love.keyboard.setKeyRepeat(true)

	app.load_config()

	if app.init_ai() then
		-- Load model async
		app.loading = true
	end
end

function love.update(dt)
	app.cursor_blink = app.cursor_blink + dt

	-- Smooth scrolling with iOS-like easing
	local diff = app.scroll_target - app.scroll_y
	if math.abs(diff) > 0.5 then
		-- Cosine easing for smooth deceleration
		local ease = 1 - math.cos((1 - math.abs(diff) / (math.abs(diff) + 100)) * math.pi / 2)
		local speed = 12 * (1 + ease * 2)
		app.scroll_y = app.scroll_y + diff * speed * dt
	else
		app.scroll_y = app.scroll_target
	end

	-- Momentum/inertia scrolling
	if math.abs(app.scroll_velocity) > 0.1 then
		app.scroll_target = app.scroll_target + app.scroll_velocity * dt
		-- Friction/deceleration
		app.scroll_velocity = app.scroll_velocity * (1 - 5 * dt)

		-- Bounce back if scrolled past bounds
		if app.scroll_target < 0 then
			app.scroll_target = app.scroll_target * 0.8
			app.scroll_velocity = app.scroll_velocity * 0.5
		end
	else
		app.scroll_velocity = 0
	end

	-- Clamp scroll
	if app.scroll_target < 0 and math.abs(app.scroll_velocity) < 1 then
		app.scroll_target = app.scroll_target * 0.9
	end

	-- Load model
	if app.loading then
		app.loading = false
		if app.load_model_async() then
			app.save_config()
		end
	end

	-- Continue generation
	if app.gen_coroutine then
		local ok, err = coroutine.resume(app.gen_coroutine)
		if not ok or coroutine.status(app.gen_coroutine) == "dead" then
			app.gen_coroutine = nil
			app.generating = false
		end
	end
end

function love.draw()
	local w, h = love.graphics.getDimensions()

	-- Background
	love.graphics.setColor(0.1, 0.1, 0.12)
	love.graphics.rectangle("fill", 0, 0, w, h)

	-- Model selection dialog
	if app.show_model_select then
		love.graphics.setColor(0.15, 0.15, 0.18)
		love.graphics.rectangle("fill", 100, 100, w - 200, h - 200)
		love.graphics.setColor(0.3, 0.3, 0.35)
		love.graphics.rectangle("line", 100, 100, w - 200, h - 200)

		love.graphics.setColor(1, 1, 1)
		love.graphics.print("Select Model", 120, 120)

		for i, name in ipairs(app.models_list) do
			local y = 150 + (i - 1) * 30
			if i == app.selected_model_idx then
				love.graphics.setColor(0.3, 0.5, 0.8)
				love.graphics.rectangle("fill", 110, y - 2, w - 220, 24)
			end
			love.graphics.setColor(1, 1, 1)
			love.graphics.print(name, 120, y)
		end

		love.graphics.setColor(0.6, 0.6, 0.6)
		love.graphics.print("↑/↓ to select, Enter to confirm", 120, h - 140)
		return
	end

	-- Messages area
	local msg_height = h - 100
	love.graphics.setScissor(0, 0, w, msg_height)

	local y = 10 - app.scroll_y
	for _, msg in ipairs(app.messages) do
		local prefix, color
		if msg.role == "user" then
			prefix = "You: "
			color = { 0.4, 0.7, 1 }
		else
			prefix = "AI: "
			color = { 0.5, 1, 0.5 }
		end

		love.graphics.setColor(color)
		local text = prefix .. msg.content
		local _, lines = app.font:getWrap(text, w - 40)
		for _, line in ipairs(lines) do
			love.graphics.print(line, 20, y)
			y = y + 30
		end
		y = y + 15
	end

	love.graphics.setScissor()

	-- Input area
	love.graphics.setColor(0.15, 0.15, 0.18)
	love.graphics.rectangle("fill", 0, h - 90, w, 90)

	-- Input box
	love.graphics.setColor(0.2, 0.2, 0.25)
	love.graphics.rectangle("fill", 10, h - 80, w - 20, 40)
	love.graphics.setColor(0.4, 0.4, 0.45)
	love.graphics.rectangle("line", 10, h - 80, w - 20, 40)

	-- Input text
	love.graphics.setColor(1, 1, 1)
	local display_text = app.input_text
	if math.floor(app.cursor_blink * 2) % 2 == 0 then
		display_text = display_text .. "|"
	end
	love.graphics.print(display_text, 20, h - 70)

	-- Status bar
	love.graphics.setFont(app.small_font)
	love.graphics.setColor(0.5, 0.5, 0.5)
	love.graphics.print(app.status, 10, h - 30)

	if app.generating then
		love.graphics.setColor(0.3, 0.8, 0.3)
		love.graphics.print("Generating...", w - 100, h - 30)
	end

	love.graphics.setFont(app.font)
end

function love.textinput(t)
	if app.show_model_select then
		return
	end
	app.input_text = app.input_text .. t
end

function love.keypressed(key)
	if app.show_model_select then
		if key == "up" then
			app.selected_model_idx = math.max(1, app.selected_model_idx - 1)
		elseif key == "down" then
			app.selected_model_idx = math.min(#app.models_list, app.selected_model_idx + 1)
		elseif key == "return" then
			app.config.model = app.models_list[app.selected_model_idx]
			app.show_model_select = false
			app.loading = true
		end
		return
	end

	if key == "return" then
		app.send_message()
	elseif key == "backspace" then
		app.input_text = app.input_text:sub(1, -2)
	elseif key == "escape" then
		love.event.quit()
	end
end

function love.wheelmoved(x, y)
	-- Add velocity for momentum scrolling (iOS-like)
	app.scroll_velocity = app.scroll_velocity - y * 800
end

function love.quit()
	app.cleanup()
end
