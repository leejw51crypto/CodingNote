# MPC Wallet (Threshold Signature Scheme)

This project implements a Multi-Party Computation (MPC) wallet using Threshold Signature Scheme (TSS) for Ethereum-compatible networks, specifically configured for Cronos testnet.

## Features

- Threshold Signature Scheme (TSS) implementation
- Support for splitting and combining private keys
- Ethereum transaction signing with multiple parties
- Configurable threshold and number of parties
- Cronos testnet integration
- Cross-language implementations (Python, Rust, Node.js)

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Rust (for Rust implementation)
- Node.js and Yarn (for Node.js implementation)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd test-mpc-wallet
```

2. Install dependencies:
```bash
# Python dependencies
pip install -r requirements.txt

# Rust dependencies (optional)
cd rusttss
cargo build
cd ..

# Node.js dependencies (optional)
cd nodetss
yarn install
cd ..
```

3. Set up environment variables:
```bash
cp .env.example .env
```

Edit `.env` file and add your private key:
```
MY_FULL_PRIVATEKEY=your_private_key_here  # Without 0x prefix
```

## Testing Vector Tutorial

The project includes comprehensive test vectors to verify TSS implementation across different languages. Here's how to run the tests:

1. Generate test vectors:
```bash
# This will create tss_test_data.json for cross-implementation testing
python tssjson.py
```

2. Run tests in different implementations:

### Python Tests
```bash
python tssjson.py
```

### Rust Tests
```bash
cd rusttss
# Run all tests including TSS implementation and test vector verification
source .env && cargo test -- --nocapture
cd ..
```

### Node.js Tests
```bash
cd nodetss
yarn test                               # Run all tests
yarn test TSSTestVector.test.ts        # Run specific test vector tests
yarn test ThresholdSignatureScheme.test.ts  # Run TSS implementation tests
cd ..
```

## Usage

The main script (`c.py`) implements TSS functionality for Ethereum transactions. To run:

```bash
source .env && python c.py
```

The script will:
1. Initialize TSS with specified threshold and number of parties
2. Generate partial signatures from each party
3. Combine signatures to create a valid transaction
4. Send the transaction to the Cronos testnet

## Gradio Web Interface

The project includes a user-friendly web interface built with Gradio (`tswallet_gradio.py`). To run the interface:

```bash
python tswallet_gradio.py
```

The Gradio interface provides:
- TSS wallet initialization with private key input
- TSS setup verification
- Transaction sending functionality
- Real-time balance checking
- User-friendly interface for interacting with the TSS wallet

⚠️ **Important**: The Gradio interface is for demonstration and educational purposes only. Never use real private keys in this interface.

## Security Considerations

- Keep your private key secure and never share it
- The `.env` file contains sensitive information - do not commit it to version control
- The threshold value determines how many parties are needed to sign a transaction
- Ensure secure communication channels between parties when sharing partial signatures

## Network Configuration

The project is configured to work with Cronos testnet:
- RPC Endpoint: https://evm-t3.cronos.org/
- Chain ID: 338

## Dependencies

### Python
- web3==6.15.1
- eth-account==0.11.0
- cryptography==42.0.5
- python-dotenv==1.0.1
- gradio==4.19.2

### Rust
See `rusttss/Cargo.toml` for Rust dependencies

### Node.js
See `nodetss/package.json` for Node.js dependencies


