use bip32::{DerivationPath, XPrv};
use bip39::{Language, Mnemonic};
use ethers::prelude::*;
use ethers::providers::{Http, Provider};
use ethers::signers::{LocalWallet, Signer};
use ethers::types::{TransactionRequest, H160, U256};
use mlua::prelude::*;
use ripemd::Ripemd160;
use secp256k1::{PublicKey, Secp256k1, SecretKey};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use sha3::{Digest, Keccak256};
use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

/// Opaque handle to wallet mnemonic - keeps sensitive data in Rust
#[derive(Clone)]
struct WalletHandle {
    mnemonic: String,
}

// Implement UserData for WalletHandle to expose to Lua
impl LuaUserData for WalletHandle {
    fn add_methods<M: LuaUserDataMethods<Self>>(methods: &mut M) {
        // Method to get mnemonic (explicit access - use sparingly)
        methods.add_method("get_mnemonic", |_, this, ()| Ok(this.mnemonic.clone()));

        // Method to get word count
        methods.add_method("word_count", |_, this, ()| {
            Ok(this.mnemonic.split_whitespace().count())
        });
    }
}

/// Generate a new mnemonic phrase - returns opaque WalletHandle
fn generate_mnemonic(_lua: &Lua, _: ()) -> LuaResult<WalletHandle> {
    use rand::RngCore;
    let mut entropy = [0u8; 32]; // 256 bits for 24 words
    rand::thread_rng().fill_bytes(&mut entropy);

    let mnemonic = Mnemonic::from_entropy_in(Language::English, &entropy)
        .map_err(|e| LuaError::RuntimeError(format!("Failed to generate mnemonic: {}", e)))?;

    Ok(WalletHandle {
        mnemonic: mnemonic.to_string(),
    })
}

/// Create WalletHandle from existing mnemonic string
fn import_mnemonic(_lua: &Lua, mnemonic_str: String) -> LuaResult<WalletHandle> {
    // Validate mnemonic
    let _mnemonic = Mnemonic::parse_in(Language::English, &mnemonic_str)
        .map_err(|e| LuaError::RuntimeError(format!("Invalid mnemonic: {}", e)))?;

    Ok(WalletHandle {
        mnemonic: mnemonic_str,
    })
}

/// Derive key at specific index using BIP32 derivation path
fn derive_key_at_index(seed: &[u8], index: u32) -> Result<SecretKey, String> {
    // Use BIP44 path for Ethereum: m/44'/60'/0'/0/{index}
    let path_str = format!("m/44'/60'/0'/0/{}", index);
    let path = DerivationPath::from_str(&path_str)
        .map_err(|e| format!("Invalid derivation path: {}", e))?;

    // Create extended private key from seed
    let xprv =
        XPrv::derive_from_path(seed, &path).map_err(|e| format!("Failed to derive key: {}", e))?;

    // Convert to secp256k1 SecretKey
    let secret_key = SecretKey::from_slice(xprv.private_key().to_bytes().as_slice())
        .map_err(|e| format!("Failed to create secret key: {}", e))?;

    Ok(secret_key)
}

/// Create wallet from WalletHandle with optional wallet index
fn create_wallet_from_mnemonic(
    _lua: &Lua,
    (handle, wallet_index): (LuaAnyUserData, Option<u32>),
) -> LuaResult<LuaTable> {
    let index = wallet_index.unwrap_or(0);

    // Extract mnemonic from handle
    let wallet_handle = handle.borrow::<WalletHandle>()?;
    let mnemonic_str = &wallet_handle.mnemonic;

    // Parse mnemonic
    let mnemonic = Mnemonic::parse_in(Language::English, mnemonic_str)
        .map_err(|e| LuaError::RuntimeError(format!("Invalid mnemonic: {}", e)))?;

    // Get seed from mnemonic
    let seed = mnemonic.to_seed("");

    // Derive key at specific index
    let secret_key = derive_key_at_index(&seed, index).map_err(|e| LuaError::RuntimeError(e))?;

    // Derive public key
    let secp = Secp256k1::new();
    let public_key = PublicKey::from_secret_key(&secp, &secret_key);

    // Generate address (Ethereum-style)
    let address = generate_eth_address(&public_key);

    // Create Lua table with wallet info (NO mnemonic in table!)
    let lua = _lua;
    let table = lua.create_table()?;
    table.set("wallet_index", index)?;
    table.set("private_key", hex::encode(secret_key.secret_bytes()))?;
    table.set(
        "public_key",
        hex::encode(public_key.serialize_uncompressed()),
    )?;
    table.set("address", address)?;

    Ok(table)
}

/// Generate multiple addresses from a WalletHandle
fn generate_addresses(
    _lua: &Lua,
    (handle, count): (LuaAnyUserData, Option<u32>),
) -> LuaResult<LuaTable> {
    let count = count.unwrap_or(5);

    // Extract mnemonic from handle
    let wallet_handle = handle.borrow::<WalletHandle>()?;
    let mnemonic_str = &wallet_handle.mnemonic;

    // Parse mnemonic
    let mnemonic = Mnemonic::parse_in(Language::English, mnemonic_str)
        .map_err(|e| LuaError::RuntimeError(format!("Invalid mnemonic: {}", e)))?;

    // Get seed from mnemonic
    let seed = mnemonic.to_seed("");
    let secp = Secp256k1::new();

    // Create Lua table to hold addresses
    let lua = _lua;
    let addresses = lua.create_table()?;

    // Generate addresses for each index
    for i in 0..count {
        let secret_key = derive_key_at_index(&seed, i).map_err(|e| LuaError::RuntimeError(e))?;

        let public_key = PublicKey::from_secret_key(&secp, &secret_key);
        let address = generate_eth_address(&public_key);

        // Create entry for this address
        let entry = lua.create_table()?;
        entry.set("index", i)?;
        entry.set("address", address)?;
        entry.set("private_key", hex::encode(secret_key.secret_bytes()))?;
        entry.set(
            "public_key",
            hex::encode(public_key.serialize_uncompressed()),
        )?;

        addresses.set(i + 1, entry)?; // Lua arrays are 1-indexed
    }

    Ok(addresses)
}

/// Generate Ethereum-style address from public key
fn generate_eth_address(public_key: &PublicKey) -> String {
    let public_key_bytes = public_key.serialize_uncompressed();

    // Skip the first byte (0x04) for uncompressed public key
    // Ethereum uses Keccak-256 (not SHA-256) for address generation
    let mut hasher = Keccak256::new();
    hasher.update(&public_key_bytes[1..]);
    let hash = hasher.finalize();

    // Take last 20 bytes
    let address = &hash[hash.len() - 20..];
    format!("0x{}", hex::encode(address))
}

/// Generate Bitcoin-style address from public key
fn generate_btc_address(_lua: &Lua, public_key_hex: String) -> LuaResult<String> {
    let public_key_bytes = hex::decode(&public_key_hex)
        .map_err(|e| LuaError::RuntimeError(format!("Invalid hex: {}", e)))?;

    let public_key = PublicKey::from_slice(&public_key_bytes)
        .map_err(|e| LuaError::RuntimeError(format!("Invalid public key: {}", e)))?;

    // SHA256 hash
    let mut sha_hasher = Sha256::new();
    sha_hasher.update(public_key.serialize());
    let sha_hash = sha_hasher.finalize();

    // RIPEMD160 hash
    let mut ripemd_hasher = Ripemd160::new();
    ripemd_hasher.update(sha_hash);
    let ripemd_hash = ripemd_hasher.finalize();

    // Add version byte (0x00 for mainnet)
    let mut address_bytes = vec![0x00];
    address_bytes.extend_from_slice(&ripemd_hash);

    // Double SHA256 for checksum
    let mut checksum_hasher = Sha256::new();
    checksum_hasher.update(&address_bytes);
    let checksum_hash1 = checksum_hasher.finalize();

    let mut checksum_hasher2 = Sha256::new();
    checksum_hasher2.update(checksum_hash1);
    let checksum_hash2 = checksum_hasher2.finalize();

    // Add first 4 bytes of checksum
    address_bytes.extend_from_slice(&checksum_hash2[0..4]);

    // Base58 encode (simplified - using hex for now)
    Ok(format!("1{}", hex::encode(&address_bytes[1..])))
}

/// OpenAI API request/response structures
#[derive(Serialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    temperature: f32,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    message: OpenAIResponseMessage,
}

#[derive(Deserialize)]
struct OpenAIResponseMessage {
    content: String,
}

#[derive(Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
}

/// Call OpenAI API to generate Lua code based on user request
fn generate_lua_code_openai(_lua: &Lua, user_request: String) -> LuaResult<String> {
    // Get API key from environment
    let api_key = env::var("OPENAI_API_KEY").map_err(|_| {
        LuaError::RuntimeError("OPENAI_API_KEY environment variable not set".to_string())
    })?;

    // Read PROTOCOL.md
    let protocol_content = std::fs::read_to_string("PROTOCOL.md")
        .unwrap_or_else(|_| "Protocol documentation not available".to_string());

    // Construct system message with protocol information
    let system_message = format!(
        "You are a Lua code generator for a wallet system. Generate ONLY executable Lua code without any markdown formatting, explanations, or comments outside the code. Do not wrap the code in ```lua blocks. Just return pure Lua code that can be executed directly.\n\nIMPORTANT API NOTES:\n- The 'wallet' module is ALREADY AVAILABLE as a global variable. DO NOT use require(\"wallet\").\n\nWALLET FUNCTIONS:\n- wallet.generate_mnemonic() returns a WalletHandle object (NOT a string)\n- To get the actual mnemonic string, call :get_mnemonic() on the handle: handle:get_mnemonic()\n- wallet.create_wallet(handle, index) takes a WalletHandle as first parameter\n- wallet.generate_addresses(handle, count) takes a WalletHandle as first parameter\n- wallet.import_mnemonic(string) creates a WalletHandle from a mnemonic string\n\nCRONOS EVM FUNCTIONS:\n- wallet.cronos_get_balance(address) - Get native token balance\n- wallet.cronos_send_tx({{private_key=..., to=..., amount=...}}) - Send native tokens (amount in wei)\n- wallet.cronos_get_tx(tx_hash) - Get transaction details\n- wallet.cronos_get_latest_block() - Get latest block number\n- wallet.cronos_get_erc20_balance({{token_address=..., address=...}}) - Get ERC20 token balance\n- wallet.cronos_send_erc20({{private_key=..., token_address=..., to=..., amount=...}}) - Send ERC20 tokens\n- wallet.get_config() - Get current RPC and chain configuration\n- wallet.set_config(table) - Update RPC and chain configuration\n\nIMPORTANT: Amount must be in WEI (not ETH). To convert: 1 ETH = 1e18 wei, 0.1 ETH = 1e17 wei\n- Use print() to display output to the user.\n- Variables you create will persist across iterations, so users can build on previous code.\n- There's a global 'current_wallet' variable that may contain a previously created wallet (access with current_wallet.address, current_wallet.private_key, etc.)\n- There's a global 'current_mnemonic_handle' variable that may contain a previously created handle (use it with wallet.create_wallet() or wallet.generate_addresses())\n- IMPORTANT: Always check if current_wallet or current_mnemonic_handle exist before using them (use: if current_wallet then ... end)\n- Keep the code simple and focused on the user's request.\n- Always use print() to show results to the user.\n\nEXAMPLE USAGE:\n```lua\n-- Example 1: Generate wallet\nlocal handle = wallet.generate_mnemonic()\nprint(\"Mnemonic: \" .. handle:get_mnemonic())\nlocal wallet_info = wallet.create_wallet(handle, 0)\nprint(\"Address: \" .. wallet_info.address)\n\n-- Example 2: Send native tokens using current_wallet\nif current_wallet then\n    -- Convert 0.1 ETH to wei\n    local amount_wei = string.format(\"%.0f\", 0.1 * 1e18)\n    local tx_hash = wallet.cronos_send_tx({{\n        private_key = current_wallet.private_key,\n        to = \"0x51aeb30cc7b31d0f5c56426f7ae8b61ba9de3a10\",\n        amount = amount_wei\n    }})\n    print(\"Transaction hash: \" .. tx_hash)\nend\n\n-- Example 3: Check balance\nif current_wallet then\n    local balance_wei = wallet.cronos_get_balance(current_wallet.address)\n    local balance_eth = tonumber(balance_wei) / 1e18\n    print(\"Balance: \" .. balance_eth .. \" ETH\")\nend\n```\n\nHere is the protocol documentation (NOTE: The protocol docs are outdated - they show old API with strings. Use the API notes above instead!):\n\n{}",
        protocol_content
    );

    let user_message = format!(
        "Generate Lua code for the following request: {}\n\nIMPORTANT REMINDERS:\n- DO NOT use require(\"wallet\") - the wallet module is already available globally\n- wallet.generate_mnemonic() returns a WalletHandle object - use :get_mnemonic() to get the string\n- wallet.create_wallet() and wallet.generate_addresses() take a WalletHandle, not a string\n- Use print() statements to show results\n- Return ONLY the Lua code, no markdown, no explanations, no code blocks.",
        user_request
    );

    // Create OpenAI request
    let request = OpenAIRequest {
        model: "gpt-4".to_string(),
        messages: vec![
            OpenAIMessage {
                role: "system".to_string(),
                content: system_message,
            },
            OpenAIMessage {
                role: "user".to_string(),
                content: user_message,
            },
        ],
        temperature: 0.7,
    };

    // Make blocking HTTP request
    let client = reqwest::blocking::Client::new();
    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .map_err(|e| LuaError::RuntimeError(format!("Failed to call OpenAI API: {}", e)))?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(LuaError::RuntimeError(format!(
            "OpenAI API error ({}): {}",
            status, error_text
        )));
    }

    // Parse response
    let openai_response: OpenAIResponse = response
        .json()
        .map_err(|e| LuaError::RuntimeError(format!("Failed to parse OpenAI response: {}", e)))?;

    // Extract generated code
    let code = openai_response
        .choices
        .first()
        .map(|choice| choice.message.content.trim().to_string())
        .ok_or_else(|| LuaError::RuntimeError("No response from OpenAI".to_string()))?;

    // Clean up code if it has markdown formatting
    let cleaned_code = if code.starts_with("```") {
        // Remove markdown code blocks
        code.lines()
            .skip(1) // Skip first ```lua line
            .take_while(|line| !line.starts_with("```"))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        code
    };

    Ok(cleaned_code)
}

/// Call Ollama API to generate Lua code based on user request
fn generate_lua_code_ollama(_lua: &Lua, user_request: String) -> LuaResult<String> {
    // Ollama runs locally, default at http://localhost:11434
    let ollama_url =
        env::var("OLLAMA_API_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());

    // Default model: gpt-oss:20b
    let model = env::var("OLLAMA_MODEL").unwrap_or_else(|_| "gpt-oss:20b".to_string());

    // Read PROTOCOL.md
    let protocol_content = std::fs::read_to_string("PROTOCOL.md")
        .unwrap_or_else(|_| "Protocol documentation not available".to_string());

    // Construct system message with protocol information
    let system_message = format!(
        "You are a Lua code generator for a wallet system. Generate ONLY executable Lua code without any markdown formatting, explanations, or comments outside the code. Do not wrap the code in ```lua blocks. Just return pure Lua code that can be executed directly.\n\nIMPORTANT API NOTES:\n- The 'wallet' module is ALREADY AVAILABLE as a global variable. DO NOT use require(\"wallet\").\n\nWALLET FUNCTIONS:\n- wallet.generate_mnemonic() returns a WalletHandle object (NOT a string)\n- To get the actual mnemonic string, call :get_mnemonic() on the handle: handle:get_mnemonic()\n- wallet.create_wallet(handle, index) takes a WalletHandle as first parameter\n- wallet.generate_addresses(handle, count) takes a WalletHandle as first parameter\n- wallet.import_mnemonic(string) creates a WalletHandle from a mnemonic string\n\nCRONOS EVM FUNCTIONS:\n- wallet.cronos_get_balance(address) - Get native token balance\n- wallet.cronos_send_tx({{private_key=..., to=..., amount=...}}) - Send native tokens (amount in wei)\n- wallet.cronos_get_tx(tx_hash) - Get transaction details\n- wallet.cronos_get_latest_block() - Get latest block number\n- wallet.cronos_get_erc20_balance({{token_address=..., address=...}}) - Get ERC20 token balance\n- wallet.cronos_send_erc20({{private_key=..., token_address=..., to=..., amount=...}}) - Send ERC20 tokens\n- wallet.get_config() - Get current RPC and chain configuration\n- wallet.set_config(table) - Update RPC and chain configuration\n\nIMPORTANT: Amount must be in WEI (not ETH). To convert: 1 ETH = 1e18 wei, 0.1 ETH = 1e17 wei\n- Use print() to display output to the user.\n- Variables you create will persist across iterations, so users can build on previous code.\n- There's a global 'current_wallet' variable that may contain a previously created wallet (access with current_wallet.address, current_wallet.private_key, etc.)\n- There's a global 'current_mnemonic_handle' variable that may contain a previously created handle (use it with wallet.create_wallet() or wallet.generate_addresses())\n- IMPORTANT: Always check if current_wallet or current_mnemonic_handle exist before using them (use: if current_wallet then ... end)\n- Keep the code simple and focused on the user's request.\n- Always use print() to show results to the user.\n\nHere is the protocol documentation (NOTE: The protocol docs are outdated - they show old API with strings. Use the API notes above instead!):\n\n{}",
        protocol_content
    );

    let user_message = format!(
        "Generate Lua code for the following request: {}\n\nIMPORTANT REMINDERS:\n- DO NOT use require(\"wallet\") - the wallet module is already available globally\n- wallet.generate_mnemonic() returns a WalletHandle object - use :get_mnemonic() to get the string\n- wallet.create_wallet() and wallet.generate_addresses() take a WalletHandle, not a string\n- Use print() statements to show results\n- Return ONLY the Lua code, no markdown, no explanations, no code blocks.",
        user_request
    );

    // Create Ollama request (compatible with OpenAI format)
    let request = OpenAIRequest {
        model: model.clone(),
        messages: vec![
            OpenAIMessage {
                role: "system".to_string(),
                content: system_message,
            },
            OpenAIMessage {
                role: "user".to_string(),
                content: user_message,
            },
        ],
        temperature: 0.7,
    };

    // Make blocking HTTP request to Ollama
    let client = reqwest::blocking::Client::new();
    let response = client
        .post(format!("{}/v1/chat/completions", ollama_url))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .map_err(|e| {
            LuaError::RuntimeError(format!(
                "Failed to call Ollama API: {}. Is Ollama running?",
                e
            ))
        })?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(LuaError::RuntimeError(format!(
            "Ollama API error ({}): {}",
            status, error_text
        )));
    }

    // Parse response
    let ollama_response: OpenAIResponse = response
        .json()
        .map_err(|e| LuaError::RuntimeError(format!("Failed to parse Ollama response: {}", e)))?;

    // Extract generated code
    let code = ollama_response
        .choices
        .first()
        .map(|choice| choice.message.content.trim().to_string())
        .ok_or_else(|| LuaError::RuntimeError("No response from Ollama".to_string()))?;

    // Clean up code if it has markdown formatting
    let cleaned_code = if code.starts_with("```") {
        // Remove markdown code blocks
        code.lines()
            .skip(1) // Skip first ```lua line
            .take_while(|line| !line.starts_with("```"))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        code
    };

    Ok(cleaned_code)
}

/// Ollama native API request structure
#[derive(Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    stream: bool,
}

/// Ollama native API response structure
#[derive(Deserialize)]
struct OllamaChatResponse {
    message: OllamaChatMessage,
}

#[derive(Deserialize)]
struct OllamaChatMessage {
    content: String,
}

/// Call Ollama Cloud API to generate Lua code based on user request
fn generate_lua_code_ollama_cloud(_lua: &Lua, user_request: String) -> LuaResult<String> {
    // Ollama Cloud endpoint
    let ollama_cloud_url = "https://ollama.com/api/chat";

    // Get API key from environment
    let api_key = env::var("OLLAMA_API_KEY").map_err(|_| {
        LuaError::RuntimeError("OLLAMA_API_KEY environment variable not set. Get your API key from https://ollama.com/settings/keys".to_string())
    })?;

    // Load model from config
    let config = load_config();
    let model = config.ollama_cloud_model;

    // Read PROTOCOL.md
    let protocol_content = std::fs::read_to_string("PROTOCOL.md")
        .unwrap_or_else(|_| "Protocol documentation not available".to_string());

    // Construct system message with protocol information
    let system_message = format!(
        "You are a Lua code generator for a wallet system. Generate ONLY executable Lua code without any markdown formatting, explanations, or comments outside the code. Do not wrap the code in ```lua blocks. Just return pure Lua code that can be executed directly.\n\nIMPORTANT API NOTES:\n- The 'wallet' module is ALREADY AVAILABLE as a global variable. DO NOT use require(\"wallet\").\n\nWALLET FUNCTIONS:\n- wallet.generate_mnemonic() returns a WalletHandle object (NOT a string)\n- To get the actual mnemonic string, call :get_mnemonic() on the handle: handle:get_mnemonic()\n- wallet.create_wallet(handle, index) takes a WalletHandle as first parameter\n- wallet.generate_addresses(handle, count) takes a WalletHandle as first parameter\n- wallet.import_mnemonic(string) creates a WalletHandle from a mnemonic string\n\nCRONOS EVM FUNCTIONS:\n- wallet.cronos_get_balance(address) - Get native token balance\n- wallet.cronos_send_tx({{private_key=..., to=..., amount=...}}) - Send native tokens (amount in wei)\n- wallet.cronos_get_tx(tx_hash) - Get transaction details\n- wallet.cronos_get_latest_block() - Get latest block number\n- wallet.cronos_get_erc20_balance({{token_address=..., address=...}}) - Get ERC20 token balance\n- wallet.cronos_send_erc20({{private_key=..., token_address=..., to=..., amount=...}}) - Send ERC20 tokens\n- wallet.get_config() - Get current RPC and chain configuration\n- wallet.set_config(table) - Update RPC and chain configuration\n\nIMPORTANT: Amount must be in WEI (not ETH). To convert: 1 ETH = 1e18 wei, 0.1 ETH = 1e17 wei\n- Use print() to display output to the user.\n- Variables you create will persist across iterations, so users can build on previous code.\n- There's a global 'current_wallet' variable that may contain a previously created wallet (access with current_wallet.address, current_wallet.private_key, etc.)\n- There's a global 'current_mnemonic_handle' variable that may contain a previously created handle (use it with wallet.create_wallet() or wallet.generate_addresses())\n- IMPORTANT: Always check if current_wallet or current_mnemonic_handle exist before using them (use: if current_wallet then ... end)\n- Keep the code simple and focused on the user's request.\n- Always use print() to show results to the user.\n\nHere is the protocol documentation (NOTE: The protocol docs are outdated - they show old API with strings. Use the API notes above instead!):\n\n{}",
        protocol_content
    );

    let user_message = format!(
        "Generate Lua code for the following request: {}\n\nIMPORTANT REMINDERS:\n- DO NOT use require(\"wallet\") - the wallet module is already available globally\n- wallet.generate_mnemonic() returns a WalletHandle object - use :get_mnemonic() to get the string\n- wallet.create_wallet() and wallet.generate_addresses() take a WalletHandle, not a string\n- Use print() statements to show results\n- Return ONLY the Lua code, no markdown, no explanations, no code blocks.",
        user_request
    );

    // Create Ollama native API request
    let request = OllamaChatRequest {
        model: model.clone(),
        messages: vec![
            OpenAIMessage {
                role: "system".to_string(),
                content: system_message,
            },
            OpenAIMessage {
                role: "user".to_string(),
                content: user_message,
            },
        ],
        stream: false,
    };

    // Make blocking HTTP request to Ollama Cloud
    let client = reqwest::blocking::Client::new();
    let response = client
        .post(ollama_cloud_url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&request)
        .send()
        .map_err(|e| LuaError::RuntimeError(format!("Failed to call Ollama Cloud API: {}", e)))?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(LuaError::RuntimeError(format!(
            "Ollama Cloud API error ({}): {}",
            status, error_text
        )));
    }

    // Parse response (Ollama native format)
    let ollama_response: OllamaChatResponse = response.json().map_err(|e| {
        LuaError::RuntimeError(format!("Failed to parse Ollama Cloud response: {}", e))
    })?;

    // Extract generated code
    let code = ollama_response.message.content.trim().to_string();

    // Clean up code if it has markdown formatting
    let cleaned_code = if code.starts_with("```") {
        // Remove markdown code blocks
        code.lines()
            .skip(1) // Skip first ```lua line
            .take_while(|line| !line.starts_with("```"))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        code
    };

    Ok(cleaned_code)
}

/// Call AI API to generate Lua code based on user request
/// Supports OpenAI, Ollama Local, and Ollama Cloud
fn generate_lua_code_with_provider(_lua: &Lua, params: (String, String)) -> LuaResult<String> {
    let (user_request, provider) = params;
    let provider = provider.to_lowercase();

    if provider == "openai" {
        generate_lua_code_openai(_lua, user_request)
    } else if provider == "ollama_cloud" {
        generate_lua_code_ollama_cloud(_lua, user_request)
    } else {
        // Default to Ollama Local
        generate_lua_code_ollama(_lua, user_request)
    }
}

/// Legacy function for backward compatibility - defaults to Ollama
fn generate_lua_code(_lua: &Lua, user_request: String) -> LuaResult<String> {
    generate_lua_code_ollama(_lua, user_request)
}

/// Get AI model configuration
fn get_ai_config(_lua: &Lua, _: ()) -> LuaResult<LuaTable> {
    let config = load_config();
    let table = _lua.create_table()?;
    table.set("ollama_cloud_model", config.ollama_cloud_model)?;
    table.set("ollama_local_model", config.ollama_local_model)?;
    Ok(table)
}

/// Set AI model configuration
fn set_ai_model(_lua: &Lua, params: (String, String)) -> LuaResult<()> {
    let (provider, model) = params;
    let mut config = load_config();

    match provider.to_lowercase().as_str() {
        "ollama_cloud" => config.ollama_cloud_model = model,
        "ollama_local" | "ollama" => config.ollama_local_model = model,
        _ => {
            return Err(LuaError::RuntimeError(format!(
                "Unknown provider: {}",
                provider
            )))
        }
    }

    save_config(&config)
        .map_err(|e| LuaError::RuntimeError(format!("Failed to save config: {}", e)))?;
    Ok(())
}

/// Response structure for Ollama tags API
#[derive(Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModelInfo>,
}

#[derive(Deserialize)]
struct OllamaModelInfo {
    name: String,
    #[serde(default)]
    size: u64,
}

/// List available Ollama Cloud models by fetching from API
fn list_ollama_cloud_models(_lua: &Lua, _: ()) -> LuaResult<LuaTable> {
    let table = _lua.create_table()?;

    // Fetch models from Ollama API
    let client = reqwest::blocking::Client::new();
    let response = client
        .get("https://ollama.com/api/tags")
        .send()
        .map_err(|e| LuaError::RuntimeError(format!("Failed to fetch Ollama models: {}", e)))?;

    if !response.status().is_success() {
        return Err(LuaError::RuntimeError(format!(
            "Failed to fetch models: HTTP {}",
            response.status()
        )));
    }

    let tags_response: OllamaTagsResponse = response
        .json()
        .map_err(|e| LuaError::RuntimeError(format!("Failed to parse models response: {}", e)))?;

    for (i, model_info) in tags_response.models.iter().enumerate() {
        let entry = _lua.create_table()?;
        entry.set("model", model_info.name.clone())?;

        // Format size in GB
        let size_gb = model_info.size as f64 / 1_000_000_000.0;
        let description = if size_gb > 0.0 {
            format!("{:.1} GB", size_gb)
        } else {
            "".to_string()
        };
        entry.set("description", description)?;
        table.set(i + 1, entry)?;
    }

    Ok(table)
}

/// Read hidden input from user (like password or mnemonic)
fn read_hidden(_lua: &Lua, prompt: String) -> LuaResult<String> {
    use std::io::Write;

    // Print prompt without newline
    print!("{}", prompt);
    std::io::stdout()
        .flush()
        .map_err(|e| LuaError::RuntimeError(format!("Failed to flush stdout: {}", e)))?;

    // Read password without echoing
    let input = rpassword::read_password()
        .map_err(|e| LuaError::RuntimeError(format!("Failed to read hidden input: {}", e)))?;

    Ok(input)
}

/// Display an image using viuer
fn display_image(
    _lua: &Lua,
    (image_path, width, height): (String, Option<u32>, Option<u32>),
) -> LuaResult<()> {
    // Check if file exists
    if !Path::new(&image_path).exists() {
        return Err(LuaError::RuntimeError(format!(
            "Image file not found: {}",
            image_path
        )));
    }

    // Use provided dimensions or defaults (in characters, not pixels)
    let width = width.unwrap_or(20); // default 20 characters
    let height = height.unwrap_or(10); // default 10 characters

    // Configure viuer with max dimensions (columns x rows)
    let config = viuer::Config {
        absolute_offset: false,
        width: Some(width),
        height: Some(height),
        ..Default::default()
    };

    // Display the image
    viuer::print_from_file(&image_path, &config)
        .map_err(|e| LuaError::RuntimeError(format!("Failed to display image: {}", e)))?;

    Ok(())
}

/// Configuration structure for info.json
#[derive(Serialize, Deserialize, Clone)]
struct WalletConfig {
    #[serde(default = "default_chain_id")]
    cronos_chain_id: u64,
    #[serde(default = "default_rpc_url")]
    cronos_rpc_url: String,
    #[serde(default)]
    default_address: Option<String>,
    #[serde(default)]
    token_addresses: std::collections::HashMap<String, String>,
    #[serde(default = "default_ollama_cloud_model")]
    ollama_cloud_model: String,
    #[serde(default = "default_ollama_local_model")]
    ollama_local_model: String,
}

fn default_chain_id() -> u64 {
    338
}

fn default_rpc_url() -> String {
    "https://evm-dev-t3.cronos.org".to_string()
}

fn default_ollama_cloud_model() -> String {
    "deepseek-v3.1:671b".to_string()
}

fn default_ollama_local_model() -> String {
    "gpt-oss:20b".to_string()
}

impl Default for WalletConfig {
    fn default() -> Self {
        Self {
            cronos_chain_id: default_chain_id(),
            cronos_rpc_url: default_rpc_url(),
            default_address: None,
            token_addresses: HashMap::new(),
            ollama_cloud_model: default_ollama_cloud_model(),
            ollama_local_model: default_ollama_local_model(),
        }
    }
}

/// Load configuration from info.json
fn load_config() -> WalletConfig {
    let config_path = "info.json";
    if Path::new(config_path).exists() {
        if let Ok(content) = std::fs::read_to_string(config_path) {
            if let Ok(config) = serde_json::from_str::<WalletConfig>(&content) {
                return config;
            }
        }
    }

    // Return default config and save it
    let config = WalletConfig::default();
    let _ = save_config(&config);
    config
}

/// Save configuration to info.json
fn save_config(config: &WalletConfig) -> Result<(), Box<dyn std::error::Error>> {
    let content = serde_json::to_string_pretty(config)?;
    std::fs::write("info.json", content)?;
    Ok(())
}

/// Get configuration as Lua table
fn get_config(_lua: &Lua, _: ()) -> LuaResult<LuaTable> {
    let config = load_config();
    let table = _lua.create_table()?;
    table.set("cronos_chain_id", config.cronos_chain_id)?;
    table.set("cronos_rpc_url", config.cronos_rpc_url.clone())?;
    if let Some(addr) = &config.default_address {
        table.set("default_address", addr.clone())?;
    }

    // Add token addresses
    let tokens_table = _lua.create_table()?;
    for (name, address) in &config.token_addresses {
        tokens_table.set(name.clone(), address.clone())?;
    }
    table.set("token_addresses", tokens_table)?;

    Ok(table)
}

/// Set configuration from Lua table
fn set_config(_lua: &Lua, table: LuaTable) -> LuaResult<()> {
    let mut config = load_config();

    if let Ok(chain_id) = table.get::<u64>("cronos_chain_id") {
        config.cronos_chain_id = chain_id;
    }
    if let Ok(rpc_url) = table.get::<String>("cronos_rpc_url") {
        config.cronos_rpc_url = rpc_url;
    }
    if let Ok(addr) = table.get::<String>("default_address") {
        config.default_address = Some(addr);
    }

    save_config(&config)
        .map_err(|e| LuaError::RuntimeError(format!("Failed to save config: {}", e)))?;
    Ok(())
}

/// Save a token address with a name
fn save_token_address(_lua: &Lua, (name, address): (String, String)) -> LuaResult<()> {
    let mut config = load_config();
    config.token_addresses.insert(name.clone(), address.clone());
    save_config(&config)
        .map_err(|e| LuaError::RuntimeError(format!("Failed to save token address: {}", e)))?;
    Ok(())
}

/// Get a token address by name
fn get_token_address(_lua: &Lua, name: String) -> LuaResult<Option<String>> {
    let config = load_config();
    Ok(config.token_addresses.get(&name).cloned())
}

/// List all saved token addresses
fn list_token_addresses(_lua: &Lua, _: ()) -> LuaResult<LuaTable> {
    let config = load_config();
    let table = _lua.create_table()?;
    for (name, address) in &config.token_addresses {
        table.set(name.clone(), address.clone())?;
    }
    Ok(table)
}

/// Helper function to create provider from config
async fn create_provider() -> Result<Provider<Http>, Box<dyn std::error::Error>> {
    let config = load_config();
    let provider = Provider::<Http>::try_from(config.cronos_rpc_url)?;
    Ok(provider)
}

/// Get native token balance for an address
fn cronos_get_balance(_lua: &Lua, address: String) -> LuaResult<String> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| LuaError::RuntimeError(format!("Failed to create runtime: {}", e)))?;

    rt.block_on(async {
        let provider = create_provider()
            .await
            .map_err(|e| LuaError::RuntimeError(format!("Failed to create provider: {}", e)))?;

        let addr: H160 = address
            .parse()
            .map_err(|e| LuaError::RuntimeError(format!("Invalid address: {}", e)))?;

        let balance = provider
            .get_balance(addr, None)
            .await
            .map_err(|e| LuaError::RuntimeError(format!("Failed to get balance: {}", e)))?;

        Ok(balance.to_string())
    })
}

/// Get latest block number
fn cronos_get_latest_block(_lua: &Lua, _: ()) -> LuaResult<u64> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| LuaError::RuntimeError(format!("Failed to create runtime: {}", e)))?;

    rt.block_on(async {
        let provider = create_provider()
            .await
            .map_err(|e| LuaError::RuntimeError(format!("Failed to create provider: {}", e)))?;

        let block_number = provider
            .get_block_number()
            .await
            .map_err(|e| LuaError::RuntimeError(format!("Failed to get block number: {}", e)))?;

        Ok(block_number.as_u64())
    })
}

/// Send native token transaction
fn cronos_send_tx(_lua: &Lua, params: LuaTable) -> LuaResult<String> {
    let private_key: String = params.get("private_key")?;
    let to_address: String = params.get("to")?;
    let amount: String = params.get("amount")?;

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| LuaError::RuntimeError(format!("Failed to create runtime: {}", e)))?;

    rt.block_on(async {
        let config = load_config();
        let provider = create_provider()
            .await
            .map_err(|e| LuaError::RuntimeError(format!("Failed to create provider: {}", e)))?;

        // Create wallet from private key
        let wallet = private_key
            .parse::<LocalWallet>()
            .map_err(|e| LuaError::RuntimeError(format!("Invalid private key: {}", e)))?
            .with_chain_id(config.cronos_chain_id);

        let client = SignerMiddleware::new(provider, wallet);

        // Parse addresses and amount
        let to: H160 = to_address
            .parse()
            .map_err(|e| LuaError::RuntimeError(format!("Invalid to address: {}", e)))?;
        let value: U256 = U256::from_dec_str(&amount)
            .map_err(|e| LuaError::RuntimeError(format!("Invalid amount: {}", e)))?;

        // Create and send transaction
        let tx = TransactionRequest::new().to(to).value(value);

        let pending_tx = client
            .send_transaction(tx, None)
            .await
            .map_err(|e| LuaError::RuntimeError(format!("Failed to send transaction: {}", e)))?;

        let tx_hash = format!("{:?}", pending_tx.tx_hash());
        Ok(tx_hash)
    })
}

/// Get transaction by hash
fn cronos_get_tx(_lua: &Lua, tx_hash: String) -> LuaResult<LuaTable> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| LuaError::RuntimeError(format!("Failed to create runtime: {}", e)))?;

    rt.block_on(async {
        let provider = create_provider()
            .await
            .map_err(|e| LuaError::RuntimeError(format!("Failed to create provider: {}", e)))?;

        let hash: H256 = tx_hash
            .parse()
            .map_err(|e| LuaError::RuntimeError(format!("Invalid tx hash: {}", e)))?;

        let tx = provider
            .get_transaction(hash)
            .await
            .map_err(|e| LuaError::RuntimeError(format!("Failed to get transaction: {}", e)))?;

        let table = _lua.create_table()?;

        if let Some(transaction) = tx {
            table.set("hash", format!("{:?}", transaction.hash))?;
            table.set("from", format!("{:?}", transaction.from))?;
            if let Some(to) = transaction.to {
                table.set("to", format!("{:?}", to))?;
            }
            table.set("value", transaction.value.to_string())?;
            table.set("gas", transaction.gas.to_string())?;
            if let Some(gas_price) = transaction.gas_price {
                table.set("gas_price", gas_price.to_string())?;
            }
            table.set("nonce", transaction.nonce.to_string())?;
            if let Some(block_number) = transaction.block_number {
                table.set("block_number", block_number.as_u64())?;
            }
        }

        Ok(table)
    })
}

/// ERC20 ABI for balance and transfer
const ERC20_ABI: &str = r#"[
    {
        "constant": true,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": false,
        "inputs": [
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    }
]"#;

/// Get ERC20 token balance
fn cronos_get_erc20_balance(_lua: &Lua, params: LuaTable) -> LuaResult<String> {
    let token_address: String = params.get("token_address")?;
    let owner_address: String = params.get("address")?;

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| LuaError::RuntimeError(format!("Failed to create runtime: {}", e)))?;

    rt.block_on(async {
        let provider = create_provider()
            .await
            .map_err(|e| LuaError::RuntimeError(format!("Failed to create provider: {}", e)))?;

        let token_addr: H160 = token_address
            .parse()
            .map_err(|e| LuaError::RuntimeError(format!("Invalid token address: {}", e)))?;
        let owner_addr: H160 = owner_address
            .parse()
            .map_err(|e| LuaError::RuntimeError(format!("Invalid owner address: {}", e)))?;

        let abi: ethers::abi::Abi = serde_json::from_str(ERC20_ABI)
            .map_err(|e| LuaError::RuntimeError(format!("Failed to parse ABI: {}", e)))?;

        let contract = Contract::new(token_addr, abi, Arc::new(provider));

        let balance: U256 = contract
            .method::<_, U256>("balanceOf", owner_addr)
            .map_err(|e| LuaError::RuntimeError(format!("Failed to create method call: {}", e)))?
            .call()
            .await
            .map_err(|e| LuaError::RuntimeError(format!("Failed to call balanceOf: {}", e)))?;

        Ok(balance.to_string())
    })
}

/// Send ERC20 token transaction
fn cronos_send_erc20(_lua: &Lua, params: LuaTable) -> LuaResult<String> {
    let private_key: String = params.get("private_key")?;
    let token_address: String = params.get("token_address")?;
    let to_address: String = params.get("to")?;
    let amount: String = params.get("amount")?;

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| LuaError::RuntimeError(format!("Failed to create runtime: {}", e)))?;

    rt.block_on(async {
        let config = load_config();
        let provider = create_provider()
            .await
            .map_err(|e| LuaError::RuntimeError(format!("Failed to create provider: {}", e)))?;

        // Create wallet from private key
        let wallet = private_key
            .parse::<LocalWallet>()
            .map_err(|e| LuaError::RuntimeError(format!("Invalid private key: {}", e)))?
            .with_chain_id(config.cronos_chain_id);

        let client = Arc::new(SignerMiddleware::new(provider, wallet));

        // Parse addresses and amount
        let token_addr: H160 = token_address
            .parse()
            .map_err(|e| LuaError::RuntimeError(format!("Invalid token address: {}", e)))?;
        let to: H160 = to_address
            .parse()
            .map_err(|e| LuaError::RuntimeError(format!("Invalid to address: {}", e)))?;
        let value: U256 = U256::from_dec_str(&amount)
            .map_err(|e| LuaError::RuntimeError(format!("Invalid amount: {}", e)))?;

        let abi: ethers::abi::Abi = serde_json::from_str(ERC20_ABI)
            .map_err(|e| LuaError::RuntimeError(format!("Failed to parse ABI: {}", e)))?;

        let contract = Contract::new(token_addr, abi, client);

        let call = contract
            .method::<_, bool>("transfer", (to, value))
            .map_err(|e| {
                LuaError::RuntimeError(format!("Failed to create transfer call: {}", e))
            })?;

        let pending_tx = call
            .send()
            .await
            .map_err(|e| LuaError::RuntimeError(format!("Failed to send transfer: {}", e)))?;

        let tx_hash = format!("{:?}", pending_tx.tx_hash());
        Ok(tx_hash)
    })
}

fn main() -> LuaResult<()> {
    let lua = Lua::new();

    // Register Rust functions to Lua
    let globals = lua.globals();

    // Create wallet module
    let wallet_module = lua.create_table()?;
    wallet_module.set("generate_mnemonic", lua.create_function(generate_mnemonic)?)?;
    wallet_module.set("import_mnemonic", lua.create_function(import_mnemonic)?)?;
    wallet_module.set(
        "create_wallet",
        lua.create_function(create_wallet_from_mnemonic)?,
    )?;
    wallet_module.set(
        "generate_addresses",
        lua.create_function(generate_addresses)?,
    )?;
    wallet_module.set(
        "generate_btc_address",
        lua.create_function(generate_btc_address)?,
    )?;
    wallet_module.set("generate_lua_code", lua.create_function(generate_lua_code)?)?;
    wallet_module.set(
        "generate_lua_code_with_provider",
        lua.create_function(generate_lua_code_with_provider)?,
    )?;
    wallet_module.set("get_ai_config", lua.create_function(get_ai_config)?)?;
    wallet_module.set("set_ai_model", lua.create_function(set_ai_model)?)?;
    wallet_module.set(
        "list_ollama_cloud_models",
        lua.create_function(list_ollama_cloud_models)?,
    )?;
    wallet_module.set("read_hidden", lua.create_function(read_hidden)?)?;
    wallet_module.set("display_image", lua.create_function(display_image)?)?;

    // Cronos EVM functions
    wallet_module.set("get_config", lua.create_function(get_config)?)?;
    wallet_module.set("set_config", lua.create_function(set_config)?)?;
    wallet_module.set(
        "save_token_address",
        lua.create_function(save_token_address)?,
    )?;
    wallet_module.set("get_token_address", lua.create_function(get_token_address)?)?;
    wallet_module.set(
        "list_token_addresses",
        lua.create_function(list_token_addresses)?,
    )?;
    wallet_module.set(
        "cronos_get_balance",
        lua.create_function(cronos_get_balance)?,
    )?;
    wallet_module.set(
        "cronos_get_latest_block",
        lua.create_function(cronos_get_latest_block)?,
    )?;
    wallet_module.set("cronos_send_tx", lua.create_function(cronos_send_tx)?)?;
    wallet_module.set("cronos_get_tx", lua.create_function(cronos_get_tx)?)?;
    wallet_module.set(
        "cronos_get_erc20_balance",
        lua.create_function(cronos_get_erc20_balance)?,
    )?;
    wallet_module.set("cronos_send_erc20", lua.create_function(cronos_send_erc20)?)?;

    globals.set("wallet", wallet_module)?;

    // Get script name from command line arguments or use default
    let args: Vec<String> = std::env::args().collect();
    let script_name = if args.len() > 1 {
        &args[1]
    } else {
        "wallet.lua"
    };

    // Load and execute the Lua script
    let lua_script = std::fs::read_to_string(script_name)
        .map_err(|e| LuaError::RuntimeError(format!("Failed to read {}: {}", script_name, e)))?;

    lua.load(&lua_script).exec()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wallet_config_default() {
        let config = WalletConfig::default();
        assert_eq!(config.cronos_chain_id, 338);
        assert_eq!(config.cronos_rpc_url, "https://evm-dev-t3.cronos.org");
        assert_eq!(config.default_address, None);
    }

    #[test]
    fn test_wallet_config_serialization() {
        let config = WalletConfig {
            cronos_chain_id: 25,
            cronos_rpc_url: "https://evm.cronos.org".to_string(),
            default_address: Some("0x1234567890123456789012345678901234567890".to_string()),
        };

        let json = serde_json::to_string(&config).expect("Failed to serialize");
        let deserialized: WalletConfig =
            serde_json::from_str(&json).expect("Failed to deserialize");

        assert_eq!(deserialized.cronos_chain_id, 25);
        assert_eq!(deserialized.cronos_rpc_url, "https://evm.cronos.org");
        assert_eq!(
            deserialized.default_address,
            Some("0x1234567890123456789012345678901234567890".to_string())
        );
    }

    #[test]
    fn test_wallet_config_save_load() {
        use std::fs;

        let test_config_path = "test_info.json";

        // Clean up if file exists
        let _ = fs::remove_file(test_config_path);

        let config = WalletConfig {
            cronos_chain_id: 338,
            cronos_rpc_url: "https://evm-dev-t3.cronos.org".to_string(),
            default_address: Some("0xabcdef0123456789abcdef0123456789abcdef01".to_string()),
        };

        // Save config
        let content = serde_json::to_string_pretty(&config).unwrap();
        fs::write(test_config_path, content).unwrap();

        // Load and verify
        let loaded_content = fs::read_to_string(test_config_path).unwrap();
        let loaded_config: WalletConfig = serde_json::from_str(&loaded_content).unwrap();

        assert_eq!(loaded_config.cronos_chain_id, config.cronos_chain_id);
        assert_eq!(loaded_config.cronos_rpc_url, config.cronos_rpc_url);
        assert_eq!(loaded_config.default_address, config.default_address);

        // Clean up
        let _ = fs::remove_file(test_config_path);
    }

    #[test]
    fn test_eth_address_generation() {
        use secp256k1::{PublicKey, Secp256k1, SecretKey};

        let secp = Secp256k1::new();
        let secret_key = SecretKey::from_slice(&[0x01; 32]).expect("Failed to create secret key");
        let public_key = PublicKey::from_secret_key(&secp, &secret_key);

        let address = generate_eth_address(&public_key);

        // Verify address format
        assert!(address.starts_with("0x"));
        assert_eq!(address.len(), 42); // "0x" + 40 hex chars
    }

    #[test]
    fn test_wallet_handle_creation() {
        let lua = Lua::new();

        // Test mnemonic generation
        let handle = generate_mnemonic(&lua, ()).expect("Failed to generate mnemonic");
        let mnemonic_str = handle.mnemonic;

        // Verify mnemonic has 24 words (for 256-bit entropy)
        let word_count = mnemonic_str.split_whitespace().count();
        assert_eq!(word_count, 24);
    }

    #[test]
    fn test_derive_key_at_index() {
        use bip39::{Language, Mnemonic};

        let test_mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
        let mnemonic =
            Mnemonic::parse_in(Language::English, test_mnemonic).expect("Failed to parse mnemonic");

        let seed = mnemonic.to_seed("");

        // Derive key at index 0
        let key_0 = derive_key_at_index(&seed, 0).expect("Failed to derive key");
        // Derive key at index 1
        let key_1 = derive_key_at_index(&seed, 1).expect("Failed to derive key");

        // Keys should be different
        assert_ne!(key_0.secret_bytes(), key_1.secret_bytes());

        // Deriving same index again should produce same key (deterministic)
        let key_0_again = derive_key_at_index(&seed, 0).expect("Failed to derive key");
        assert_eq!(key_0.secret_bytes(), key_0_again.secret_bytes());
    }

    #[test]
    fn test_erc20_abi_parsing() {
        let abi: Result<ethers::abi::Abi, _> = serde_json::from_str(ERC20_ABI);
        assert!(abi.is_ok(), "ERC20 ABI should parse correctly");

        let parsed_abi = abi.unwrap();
        assert!(
            parsed_abi.functions().any(|f| f.name == "balanceOf"),
            "ABI should contain balanceOf function"
        );
        assert!(
            parsed_abi.functions().any(|f| f.name == "transfer"),
            "ABI should contain transfer function"
        );
    }
}
