# Lua ECDSA Wallet Manager

A hierarchical deterministic (HD) wallet manager with Lua scripting support, built with Rust for cryptographic operations and Cronos EVM blockchain integration.

## Features

- **HD Wallet Support**: BIP32/BIP39/BIP44 compliant wallet generation
- **Multi-Chain Compatible**: Ethereum-style address generation
- **Cronos EVM Integration**: Native token and ERC20 token support
- **AI-Powered Scripting**: Generate and execute Lua code using OpenAI or Ollama (local)
- **Session Management**: Login/logout functionality with persistent wallet sessions
- **Secure Key Management**: Hidden input for sensitive data
- **Interactive CLI**: Menu-driven interface for all wallet operations

## Prerequisites

- Rust 1.70+ and Cargo
- **For AI features (optional)**:
  - OpenAI API key, OR
  - [Ollama](https://ollama.ai/) installed locally with `gpt-oss:20b` model

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd test-lua
```

2. Build the project:
```bash
cargo build --release
```

3. Run the wallet:
```bash
cargo run --release
```

## Configuration

The wallet creates an `info.json` file with default configuration:

```json
{
  "cronos_chain_id": 338,
  "cronos_rpc_url": "https://evm-dev-t3.cronos.org",
  "default_address": null,
  "token_addresses": {}
}
```

### RPC Configuration

- **Default**: Cronos Testnet (Chain ID: 338)
- **Customize**: Use menu option 15 to change RPC URL and chain ID

### AI Features Setup

**Option 1: Ollama (Local AI - Recommended)**

1. Install Ollama: https://ollama.ai/
2. Pull the model:
```bash
ollama pull gpt-oss:20b
```
3. No API key needed - runs locally!
4. Use menu option `9` for Ollama AI mode

**Option 2: OpenAI (Cloud AI)**

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file (make sure it's in `.gitignore`).
Use menu option `91` for OpenAI AI mode.

## Usage

### Quick Start

**First Time Login:**

1. **Generate a new wallet**:
   - Select option `1` to generate a mnemonic
   - Save the 24-word phrase securely
   - Create wallet at index 0
   - **Wallet session is automatically saved** to `wallet_session.json`

2. **Import existing wallet**:
   - Select option `2`
   - Enter your mnemonic phrase (input is hidden)
   - Specify wallet index (default: 0)
   - **Wallet session is automatically saved**

**Next Time:**
- Your wallet session persists - just run `cargo run` and you'll be logged in automatically
- Use option `19` to logout and clear the session

**Using Your Wallet:**

3. **Check balance**:
   - Select option `10` for native token balance
   - Select option `14` for ERC20 token balance

4. **Send transactions**:
   - Select option `11` for native tokens
   - Select option `15` for ERC20 tokens

### Menu Options

**When Logged In:**
```
1.  Generate New Mnemonic
2.  Create Wallet from Mnemonic
3.  Show Wallet Information
4.  Show Private Key
5.  Show Public Key
6.  Show Addresses (Multiple)
7.  Show Current Address
8.  Show Addresses & Keys (Detailed view with private keys)
9.  AI - Ollama (Local AI)
10. Get Balance (Native Token)
11. Send Transaction (Native Token)
12. Get Transaction by Hash
13. Get Latest Block Number
14. Get ERC20 Token Balance
15. Send ERC20 Token
16. View/Update Config (RPC, Chain ID)
17. Set Default Address
18. Toggle AI Autorun
19. Logout
91. AI - OpenAI (Cloud AI)
0.  Exit
```

**When Not Logged In:**
```
1. Create New Wallet
2. Import Existing Wallet
0. Exit
```

### BIP44 Derivation Path

The wallet uses the standard Ethereum derivation path:
```
m/44'/60'/0'/0/{index}
```

### Amount Format

When sending transactions, amounts must be in **wei** (not ETH):
- 1 ETH = 1,000,000,000,000,000,000 wei (1e18)
- 0.1 ETH = 100,000,000,000,000,000 wei (1e17)

You can use scientific notation: `1e18` for 1 token.

## AI Mode

The wallet supports two AI providers:

- **Option 9 - Ollama (Local)**: Runs AI models locally on your machine (default: `gpt-oss:20b`)
- **Option 91 - OpenAI (Cloud)**: Uses OpenAI's GPT-4 API (requires API key)

Both modes generate Lua code based on natural language requests:

### Examples:

```
AI> show my current wallet balance
AI> send 0.1 tokens to 0x51aeb30cc7b31d0f5c56426f7ae8b61ba9de3a10
AI> generate 10 addresses from my mnemonic
AI> check the latest block number
```

### AI Autorun Mode

- **Enabled**: Code executes automatically without confirmation
- **Disabled**: You'll be prompted to review code before execution
- Toggle with menu option `17` or type `toggle` in AI mode

### Available Global Variables in AI Mode

- `current_wallet` - Currently loaded wallet (address, private_key, etc.)
- `current_mnemonic_handle` - Current mnemonic handle
- `wallet` - Module with all wallet functions

## Security

### Important Warnings

- **Never share your mnemonic phrase** - Anyone with your mnemonic can access your funds
- **Never share your private keys** - They provide full control of your wallet
- **Backup your mnemonic** - Store it securely offline
- **Use testnet first** - Test with testnet tokens before using real funds

### Files to Keep Private

The `.gitignore` is configured to exclude:
- `*.wallet` - Wallet data files
- `*.keystore` - Keystore files
- `wallet_session.json` - **Contains private keys and mnemonics in plaintext!**
- `info.json` - Configuration with addresses
- `.env` - Environment variables

**Never commit these files to version control!**

⚠️ **CRITICAL**: The `wallet_session.json` file stores your private keys unencrypted for convenience. This is acceptable ONLY for:
- Educational/testing purposes
- Testnet wallets with no real value
- Local development environments

**NEVER** use this wallet with real funds or on production systems!

### Best Practices

1. Use environment variables for API keys
2. Test on testnets before mainnet
3. Verify recipient addresses before sending
4. Keep small amounts in hot wallets
5. Use hardware wallets for large amounts

## Development

### Project Structure

```
test-lua/
├── src/
│   └── main.rs          # Rust backend with crypto functions
├── wallet.lua           # Lua frontend interface
├── Cargo.toml          # Rust dependencies
├── .gitignore          # Git exclusions
└── info.json           # Config (auto-generated, gitignored)
```

### Adding Custom Lua Functions

The Rust backend exposes functions to Lua via the `wallet` module:

```lua
-- Example: Using wallet functions
local handle = wallet.generate_mnemonic()
local wallet_info = wallet.create_wallet(handle, 0)
print(wallet_info.address)
```

### Extending Blockchain Support

To add support for other EVM chains:

1. Update `cronos_chain_id` in `info.json`
2. Update `cronos_rpc_url` to your chain's RPC endpoint
3. Use menu option 15 to configure

## Troubleshooting

### "OPENAI_API_KEY environment variable not set"
If using OpenAI (option 91), set your API key:
```bash
export OPENAI_API_KEY="sk-..."
```

Alternatively, use Ollama (option 9) which doesn't require an API key.

### "Failed to call Ollama API: Connection refused"
Ensure Ollama is running:
```bash
# Start Ollama service
ollama serve

# In another terminal, verify it's running
ollama list
```

### "Failed to create provider"
Check your RPC URL and internet connection. Verify the RPC endpoint is accessible:
```bash
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
  https://evm-dev-t3.cronos.org
```

### Build Errors
Ensure you have the latest Rust toolchain:
```bash
rustup update
cargo clean
cargo build --release
```

## Token Address Management

Save frequently used token addresses:

1. When checking balance or sending tokens, you'll be prompted to save
2. Saved tokens can be referenced by name instead of address
3. View saved tokens in menu option 15

Example:
```
Enter token contract address: USDT
# Uses saved address instead of typing full address
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Test thoroughly on testnet
4. Submit a pull request

## Disclaimer

This software is provided "as is" without warranty of any kind. Use at your own risk.

- **Not audited**: This code has not undergone a professional security audit
- **Testnet recommended**: Use testnets for testing and learning
- **No liability**: Authors are not responsible for loss of funds

## License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

## Resources

- [BIP39 Specification](https://github.com/bitcoin/bips/blob/master/bip-0039.mediawiki)
- [BIP32 Specification](https://github.com/bitcoin/bips/blob/master/bip-0032.mediawiki)
- [BIP44 Specification](https://github.com/bitcoin/bips/blob/master/bip-0044.mediawiki)
- [Cronos Documentation](https://docs.cronos.org/)
- [Ethereum JSON-RPC](https://ethereum.org/en/developers/docs/apis/json-rpc/)
