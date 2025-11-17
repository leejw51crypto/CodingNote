# Lua ECDSA Wallet Manager

A hierarchical deterministic (HD) wallet manager with Lua scripting support, built with Rust for cryptographic operations and Cronos EVM blockchain integration.

## Features

- **HD Wallet Support**: BIP32/BIP39/BIP44 compliant wallet generation
- **Multi-Chain Compatible**: Ethereum-style address generation
- **Cronos EVM Integration**: Native token and ERC20 token support
- **AI-Powered Scripting**: Generate and execute Lua code using OpenAI
- **Secure Key Management**: Hidden input for sensitive data
- **Interactive CLI**: Menu-driven interface for all wallet operations

## Prerequisites

- Rust 1.70+ and Cargo
- OpenAI API key (optional, only needed for AI features)

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

To use AI code generation (menu option 8):

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file (make sure it's in `.gitignore`).

## Usage

### Quick Start

1. **Generate a new wallet**:
   - Select option `1` to generate a mnemonic
   - Save the 24-word phrase securely
   - Create wallet at index 0

2. **Import existing wallet**:
   - Select option `2`
   - Enter your mnemonic phrase (input is hidden)
   - Specify wallet index (default: 0)

3. **Check balance**:
   - Select option `9` for native token balance
   - Select option `13` for ERC20 token balance

4. **Send transactions**:
   - Select option `10` for native tokens
   - Select option `14` for ERC20 tokens

### Menu Options

```
1.  Generate New Mnemonic
2.  Create Wallet from Mnemonic
3.  Show Wallet Information
4.  Show Private Key
5.  Show Public Key
6.  Show Addresses (Multiple)
7.  Show Current Address
8.  AI - Generate and Run Lua Code
9.  Get Balance (Native Token)
10. Send Transaction (Native Token)
11. Get Transaction by Hash
12. Get Latest Block Number
13. Get ERC20 Token Balance
14. Send ERC20 Token
15. View/Update Config (RPC, Chain ID)
16. Set Default Address
17. Toggle AI Autorun
0.  Exit
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

The AI mode (option 8) generates Lua code based on natural language requests:

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
- `info.json` - Configuration with addresses
- `.env` - Environment variables

**Never commit these files to version control!**

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
Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
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
