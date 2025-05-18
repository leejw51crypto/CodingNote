# Blockchain Simulation with Bevy

A real-time blockchain simulation built with Rust and Bevy game engine. This project demonstrates core blockchain concepts including mining, transactions, and network dynamics.

## Features

- 🔗 Real-time block generation and mining
- 💰 Dynamic transaction generation between participants
- ⚡ Adjustable mining difficulty
- 📊 Network hashrate simulation
- 🏢 Company and participant simulation
- 💎 Mining rewards with halving mechanism
- 📈 Detailed statistics and reporting

## Prerequisites

- Rust (latest stable version)
- Cargo (Rust's package manager)

## Dependencies

```toml
[dependencies]
bevy = "0.12" # Game engine
chrono = "0.4" # DateTime handling
fake = "2.9" # Fake data generation
rand = "0.8" # Random number generation
sha2 = "0.10" # SHA256 hashing
```

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/blockchain-simulation
cd blockchain-simulation
```

2. Run the simulation:
```bash
cargo run
```

## How It Works

The simulation creates a blockchain environment where:

1. Random participants (companies) are generated with unique names and industries
2. Transactions are automatically generated between participants
3. Blocks are mined with a proof-of-work mechanism
4. Mining rewards are distributed to successful miners
5. Network hashrate increases over time
6. Detailed statistics are shown after completion

## Output

The simulation provides real-time feedback including:
- Block mining progress
- Transaction details
- Mining rewards
- Network statistics
- Participant activity

## Configuration

Key parameters can be adjusted in `main.rs`:
- `blocks_to_generate`: Number of blocks to simulate
- `difficulty`: Mining difficulty
- `network_hashrate`: Initial network hashrate
- Number of participants and transaction frequency

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Feel free to submit issues and pull requests. 