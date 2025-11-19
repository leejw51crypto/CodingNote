-- llama.lua - LuaJIT FFI bindings for llama.cpp
-- Requires LuaJIT with FFI support

local ffi = require("ffi")

-- Define llama.cpp C API types and functions
ffi.cdef([[
    // Basic types
    typedef int32_t llama_token;
    typedef int32_t llama_seq_id;
    typedef int32_t llama_pos;

    // Opaque types
    typedef struct llama_model llama_model;
    typedef struct llama_context llama_context;
    typedef struct llama_vocab llama_vocab;
    typedef struct llama_sampler llama_sampler;

    // Model parameters - must match llama.h exactly
    typedef struct llama_model_params {
        void * devices;
        void * tensor_buft_overrides;
        int32_t n_gpu_layers;
        int32_t split_mode;
        int32_t main_gpu;
        const float * tensor_split;
        void * progress_callback;
        void * progress_callback_user_data;
        void * kv_overrides;
        bool vocab_only;
        bool use_mmap;
        bool use_mlock;
        bool check_tensors;
    } llama_model_params;

    // Context parameters - must match llama.h exactly
    typedef struct llama_context_params {
        uint32_t n_ctx;
        uint32_t n_batch;
        uint32_t n_ubatch;
        uint32_t n_seq_max;
        int32_t n_threads;
        int32_t n_threads_batch;
        int32_t rope_scaling_type;
        int32_t pooling_type;
        int32_t attention_type;
        int32_t flash_attn_type;
        float rope_freq_base;
        float rope_freq_scale;
        float yarn_ext_factor;
        float yarn_attn_factor;
        float yarn_beta_fast;
        float yarn_beta_slow;
        uint32_t yarn_orig_ctx;
        float defrag_thold;
        void * cb_eval;
        void * cb_eval_user_data;
        int32_t type_k;
        int32_t type_v;
        void * abort_callback;
        void * abort_callback_data;
        bool embeddings;
        bool offload_kqv;
        bool no_perf;
        bool op_offload;
        bool swa_full;
        bool kv_unified;
    } llama_context_params;

    // Batch structure
    typedef struct llama_batch {
        int32_t n_tokens;
        llama_token * token;
        float * embd;
        llama_pos * pos;
        int32_t * n_seq_id;
        llama_seq_id ** seq_id;
        int8_t * logits;
    } llama_batch;

    // Sampler chain params
    typedef struct llama_sampler_chain_params {
        bool no_perf;
    } llama_sampler_chain_params;

    // Core functions
    void llama_backend_init(void);
    void llama_backend_free(void);

    // Model functions
    struct llama_model_params llama_model_default_params(void);
    struct llama_model * llama_model_load_from_file(const char * path_model, struct llama_model_params params);
    void llama_model_free(struct llama_model * model);

    // Context functions
    struct llama_context_params llama_context_default_params(void);
    struct llama_context * llama_init_from_model(struct llama_model * model, struct llama_context_params params);
    void llama_free(struct llama_context * ctx);
    uint32_t llama_n_ctx(const struct llama_context * ctx);
    uint32_t llama_n_batch(const struct llama_context * ctx);

    // Vocab functions
    const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model);
    int32_t llama_vocab_n_tokens(const struct llama_vocab * vocab);
    llama_token llama_vocab_bos(const struct llama_vocab * vocab);
    llama_token llama_vocab_eos(const struct llama_vocab * vocab);
    bool llama_vocab_is_eog(const struct llama_vocab * vocab, llama_token token);

    // Tokenization
    int32_t llama_tokenize(
        const struct llama_vocab * vocab,
        const char * text,
        int32_t text_len,
        llama_token * tokens,
        int32_t n_tokens_max,
        bool add_special,
        bool parse_special
    );

    int32_t llama_token_to_piece(
        const struct llama_vocab * vocab,
        llama_token token,
        char * buf,
        int32_t length,
        int32_t lstrip,
        bool special
    );

    int32_t llama_detokenize(
        const struct llama_vocab * vocab,
        const llama_token * tokens,
        int32_t n_tokens,
        char * text,
        int32_t text_len_max,
        bool remove_special,
        bool unparse_special
    );

    // Batch functions
    struct llama_batch llama_batch_get_one(llama_token * tokens, int32_t n_tokens);
    struct llama_batch llama_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max);
    void llama_batch_free(struct llama_batch batch);

    // Inference
    int32_t llama_decode(struct llama_context * ctx, struct llama_batch batch);
    float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);

    // Memory/KV cache
    typedef struct llama_memory_i * llama_memory_t;
    llama_memory_t llama_get_memory(const struct llama_context * ctx);
    void llama_memory_clear(llama_memory_t mem, bool data);

    // Sampler functions
    struct llama_sampler_chain_params llama_sampler_chain_default_params(void);
    struct llama_sampler * llama_sampler_chain_init(struct llama_sampler_chain_params params);
    void llama_sampler_chain_add(struct llama_sampler * chain, struct llama_sampler * smpl);
    llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx);
    void llama_sampler_accept(struct llama_sampler * smpl, llama_token token);
    void llama_sampler_reset(struct llama_sampler * smpl);
    void llama_sampler_free(struct llama_sampler * smpl);

    // Sampler initializers
    struct llama_sampler * llama_sampler_init_greedy(void);
    struct llama_sampler * llama_sampler_init_dist(uint32_t seed);
    struct llama_sampler * llama_sampler_init_top_k(int32_t k);
    struct llama_sampler * llama_sampler_init_top_p(float p, size_t min_keep);
    struct llama_sampler * llama_sampler_init_min_p(float p, size_t min_keep);
    struct llama_sampler * llama_sampler_init_temp(float t);
    struct llama_sampler * llama_sampler_init_penalties(
        int32_t penalty_last_n,
        float penalty_repeat,
        float penalty_freq,
        float penalty_present
    );

    // Model info
    int32_t llama_model_n_ctx_train(const struct llama_model * model);
    int32_t llama_model_n_embd(const struct llama_model * model);
]])

-- Load the llama.cpp shared library
local lib_path = os.getenv("LLAMA_LIB_PATH")
if not lib_path then
	local os_name = ffi.os
	if os_name == "OSX" then
		lib_path = "libllama.dylib"
	elseif os_name == "Windows" then
		lib_path = "llama.dll"
	else
		lib_path = "libllama.so"
	end
end
local llama = ffi.load(lib_path)

-- Llama module
local M = {}

-- Initialize backend (call once)
function M.init()
	llama.llama_backend_init()
end

-- Cleanup backend (call once)
function M.cleanup()
	llama.llama_backend_free()
end

-- Model class
local Model = {}
Model.__index = Model

function M.load_model(model_path, params)
	local model_params = llama.llama_model_default_params()

	if params then
		if params.n_gpu_layers then
			model_params.n_gpu_layers = params.n_gpu_layers
		end
		if params.use_mmap ~= nil then
			model_params.use_mmap = params.use_mmap
		end
		if params.use_mlock ~= nil then
			model_params.use_mlock = params.use_mlock
		end
	end

	local model_ptr = llama.llama_model_load_from_file(model_path, model_params)
	if model_ptr == nil then
		return nil, "Failed to load model: " .. model_path
	end

	local self = setmetatable({}, Model)
	self._model = model_ptr
	self._vocab = llama.llama_model_get_vocab(model_ptr)

	return self
end

function Model:free()
	if self._model then
		llama.llama_model_free(self._model)
		self._model = nil
		self._vocab = nil
	end
end

function Model:bos_token()
	return llama.llama_vocab_bos(self._vocab)
end

function Model:eos_token()
	return llama.llama_vocab_eos(self._vocab)
end

function Model:is_eog(token)
	return llama.llama_vocab_is_eog(self._vocab, token)
end

function Model:vocab_size()
	return llama.llama_vocab_n_tokens(self._vocab)
end

function Model:tokenize(text, add_special, parse_special)
	add_special = add_special ~= false
	parse_special = parse_special ~= false

	local text_len = #text
	local max_tokens = text_len + 128
	local tokens = ffi.new("llama_token[?]", max_tokens)

	local n_tokens = llama.llama_tokenize(self._vocab, text, text_len, tokens, max_tokens, add_special, parse_special)

	if n_tokens < 0 then
		max_tokens = -n_tokens
		tokens = ffi.new("llama_token[?]", max_tokens)
		n_tokens = llama.llama_tokenize(self._vocab, text, text_len, tokens, max_tokens, add_special, parse_special)
	end

	local result = {}
	for i = 0, n_tokens - 1 do
		table.insert(result, tokens[i])
	end

	return result, tokens, n_tokens
end

function Model:token_to_piece(token)
	local buf = ffi.new("char[?]", 256)
	local len = llama.llama_token_to_piece(self._vocab, token, buf, 256, 0, true)
	if len < 0 then
		return ""
	end
	return ffi.string(buf, len)
end

function Model:detokenize(tokens)
	local n_tokens = #tokens
	local tokens_arr = ffi.new("llama_token[?]", n_tokens)
	for i = 1, n_tokens do
		tokens_arr[i - 1] = tokens[i]
	end

	local buf_size = n_tokens * 8
	local buf = ffi.new("char[?]", buf_size)

	local len = llama.llama_detokenize(self._vocab, tokens_arr, n_tokens, buf, buf_size, false, true)

	if len < 0 then
		return ""
	end
	return ffi.string(buf, len)
end

-- Context class
local Context = {}
Context.__index = Context

function M.create_context(model, params)
	local ctx_params = llama.llama_context_default_params()

	if params then
		if params.n_ctx then
			ctx_params.n_ctx = params.n_ctx
		end
		if params.n_batch then
			ctx_params.n_batch = params.n_batch
		end
		if params.n_threads then
			ctx_params.n_threads = params.n_threads
			ctx_params.n_threads_batch = params.n_threads
		end
		if params.embeddings ~= nil then
			ctx_params.embeddings = params.embeddings
		end
	end

	local ctx_ptr = llama.llama_init_from_model(model._model, ctx_params)
	if ctx_ptr == nil then
		return nil, "Failed to create context"
	end

	local self = setmetatable({}, Context)
	self._ctx = ctx_ptr
	self._model = model

	return self
end

function Context:free()
	if self._ctx then
		llama.llama_free(self._ctx)
		self._ctx = nil
	end
end

function Context:n_ctx()
	return llama.llama_n_ctx(self._ctx)
end

function Context:clear_kv_cache()
	local mem = llama.llama_get_memory(self._ctx)
	llama.llama_memory_clear(mem, false)
end

function Context:decode(tokens)
	local n_tokens = #tokens
	local n_batch = llama.llama_n_batch(self._ctx)

	-- Process tokens in batches to avoid exceeding n_batch limit
	local pos = 1
	while pos <= n_tokens do
		local batch_size = math.min(n_batch, n_tokens - pos + 1)
		local tokens_arr = ffi.new("llama_token[?]", batch_size)
		for i = 0, batch_size - 1 do
			tokens_arr[i] = tokens[pos + i]
		end

		local batch = llama.llama_batch_get_one(tokens_arr, batch_size)
		if llama.llama_decode(self._ctx, batch) ~= 0 then
			return false
		end
		pos = pos + batch_size
	end

	return true
end

-- Sampler class
local Sampler = {}
Sampler.__index = Sampler

function M.create_sampler(params)
	params = params or {}

	local chain_params = llama.llama_sampler_chain_default_params()
	local chain = llama.llama_sampler_chain_init(chain_params)

	if params.top_k and params.top_k > 0 then
		llama.llama_sampler_chain_add(chain, llama.llama_sampler_init_top_k(params.top_k))
	end

	if params.top_p and params.top_p < 1.0 then
		llama.llama_sampler_chain_add(chain, llama.llama_sampler_init_top_p(params.top_p, 1))
	end

	if params.min_p and params.min_p > 0.0 then
		llama.llama_sampler_chain_add(chain, llama.llama_sampler_init_min_p(params.min_p, 1))
	end

	if params.temperature then
		if params.temperature == 0 then
			llama.llama_sampler_chain_add(chain, llama.llama_sampler_init_greedy())
		else
			llama.llama_sampler_chain_add(chain, llama.llama_sampler_init_temp(params.temperature))
			llama.llama_sampler_chain_add(chain, llama.llama_sampler_init_dist(params.seed or 0))
		end
	else
		llama.llama_sampler_chain_add(chain, llama.llama_sampler_init_greedy())
	end

	if params.repeat_penalty and params.repeat_penalty ~= 1.0 then
		llama.llama_sampler_chain_add(
			chain,
			llama.llama_sampler_init_penalties(
				params.penalty_last_n or 64,
				params.repeat_penalty,
				params.freq_penalty or 0.0,
				params.presence_penalty or 0.0
			)
		)
	end

	local self = setmetatable({}, Sampler)
	self._sampler = chain
	return self
end

function Sampler:free()
	if self._sampler then
		llama.llama_sampler_free(self._sampler)
		self._sampler = nil
	end
end

function Sampler:sample(ctx, idx)
	return llama.llama_sampler_sample(self._sampler, ctx._ctx, idx or -1)
end

function Sampler:accept(token)
	llama.llama_sampler_accept(self._sampler, token)
end

function Sampler:reset()
	llama.llama_sampler_reset(self._sampler)
end

-- High-level generation function
function M.generate(model, ctx, sampler, prompt, max_tokens, callback)
	max_tokens = max_tokens or 128

	-- Clear KV cache before each generation to prevent overflow
	ctx:clear_kv_cache()

	local tokens = model:tokenize(prompt)
	if not ctx:decode(tokens) then
		return nil, "Failed to decode prompt"
	end

	local generated = {}

	for _ = 1, max_tokens do
		local new_token = sampler:sample(ctx, -1)

		if model:is_eog(new_token) then
			break
		end

		sampler:accept(new_token)
		table.insert(generated, new_token)

		if callback then
			callback(model:token_to_piece(new_token), new_token)
		end

		if not ctx:decode({ new_token }) then
			return nil, "Failed to decode token"
		end
	end

	return model:detokenize(generated), generated
end

return M
