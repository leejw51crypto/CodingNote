import random
import secrets
import tiktoken
from typing import List, Tuple


def generate_random_wallet_address() -> str:
    """Generate a random Ethereum wallet address (0x + 40 hex chars)"""
    # Generate 20 random bytes (40 hex characters)
    address_bytes = secrets.token_bytes(20)
    # Convert to hex and add 0x prefix
    return "0x" + address_bytes.hex()


def generate_multiple_addresses(count: int = 5) -> List[str]:
    """Generate multiple random wallet addresses"""
    return [generate_random_wallet_address() for _ in range(count)]


def analyze_tokenization(text: str) -> Tuple[List[int], List[str], int]:
    """
    Analyze how GPT tokenizes the given text using cl100k_base encoding
    Returns: (token_ids, token_strings, token_count)
    """
    # Get the cl100k_base encoding (used by GPT-4 and GPT-3.5-turbo)
    encoding = tiktoken.get_encoding("cl100k_base")

    # Encode the text to get token IDs
    token_ids = encoding.encode(text)

    # Decode each token ID to see the actual token strings
    token_strings = [encoding.decode([token_id]) for token_id in token_ids]

    # Count tokens
    token_count = len(token_ids)

    return token_ids, token_strings, token_count


def display_tokenization_analysis(text: str, description: str = ""):
    """Display detailed tokenization analysis for given text"""
    print(f"\n{'='*70}")
    print(f"TOKENIZATION ANALYSIS: {description}")
    print(f"{'='*70}")
    print(f"Original text: {text}")
    print(f"Text length: {len(text)} characters")

    try:
        token_ids, token_strings, token_count = analyze_tokenization(text)

        print(f"\n🤖 Encoding: cl100k_base (GPT-4/GPT-3.5-turbo)")
        print(f"📊 Token count: {token_count}")
        print(f"💰 Compression ratio: {len(text)/token_count:.2f} chars per token")

        print(f"\n🔤 Token breakdown:")
        for i, (token_id, token_str) in enumerate(zip(token_ids, token_strings)):
            # Show token with visual boundaries and escape special chars
            token_display = repr(token_str)
            print(f"  [{i+1:2d}] ID:{token_id:5d} → {token_display}")

        print(f"\n📝 Reconstructed: {''.join(token_strings)}")

    except Exception as e:
        print(f"❌ Error: {e}")


def compare_address_formats():
    """Compare tokenization of different wallet address formats"""

    # Generate sample addresses
    addresses = generate_multiple_addresses(3)

    print("🚀 GPT TOKENIZATION ANALYSIS FOR WALLET ADDRESSES")
    print("=" * 70)
    print("This shows how GPT breaks down wallet addresses into tokens")
    print("Understanding tokenization helps optimize context usage!\n")

    # Analyze individual addresses
    for i, address in enumerate(addresses, 1):
        display_tokenization_analysis(address, f"Wallet Address #{i}")

    # Analyze multiple addresses in one string
    multiple_addresses = " ".join(addresses)
    display_tokenization_analysis(multiple_addresses, "Multiple Addresses")

    # Analyze addresses with labels
    labeled_addresses = f"Main wallet: {addresses[0]}, Trading: {addresses[1]}, Cold storage: {addresses[2]}"
    display_tokenization_analysis(labeled_addresses, "Labeled Addresses")

    # Analyze common patterns
    patterns = [
        "0x" + "0" * 40,  # All zeros
        "0x" + "f" * 40,  # All F's
        "0x" + "1234567890abcdef" * 2 + "12345678",  # Pattern
        addresses[0].upper(),  # Uppercase
        addresses[0].lower(),  # Lowercase (already lowercase but for comparison)
    ]

    print(f"\n{'='*70}")
    print("PATTERN COMPARISON")
    print(f"{'='*70}")

    pattern_names = [
        "All Zeros",
        "All F's",
        "Repeating Pattern",
        "Uppercase",
        "Lowercase",
    ]

    for pattern, name in zip(patterns, pattern_names):
        _, _, token_count = analyze_tokenization(pattern)
        print(f"{name:20} | {pattern[:20]}... | {token_count:2d} tokens")


def interactive_tokenizer():
    """Interactive mode to analyze custom wallet addresses"""
    print(f"\n{'='*70}")
    print("INTERACTIVE TOKENIZER")
    print(f"{'='*70}")
    print("Enter wallet addresses to see how GPT tokenizes them!")
    print("Type 'quit' to exit, 'random' to generate a random address")

    while True:
        user_input = input("\n🔤 Enter wallet address (or 'quit'/'random'): ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("👋 Goodbye!")
            break
        elif user_input.lower() in ["random", "r"]:
            address = generate_random_wallet_address()
            print(f"🎲 Generated random address: {address}")
            display_tokenization_analysis(address, "Random Generated Address")
        elif user_input.startswith("0x") and len(user_input) == 42:
            display_tokenization_analysis(user_input, "User Provided Address")
        else:
            if user_input:
                print(
                    "⚠️  Invalid wallet address format. Should be 0x followed by 40 hex characters"
                )
                print("💡 Example: 0x742d35Cc6A7FbC4e3fA8D5F3B8e2C9A1D4E7F8B2")


def token_cost_analysis():
    """Analyze token costs for wallet address storage"""
    print(f"\n{'='*70}")
    print("TOKEN COST ANALYSIS")
    print(f"{'='*70}")

    # Sample addresses for cost analysis
    addresses = generate_multiple_addresses(10)

    single_address = addresses[0]
    multiple_context = "\n".join(
        [f"Wallet {i+1}: {addr}" for i, addr in enumerate(addresses)]
    )

    scenarios = [
        ("Single Address", single_address),
        ("10 Addresses (labeled)", multiple_context),
        ("JSON Format", f'{{"wallets": {addresses}}}'),
        (
            "CSV Format",
            "address,label\n"
            + "\n".join([f"{addr},Wallet{i+1}" for i, addr in enumerate(addresses)]),
        ),
    ]

    print("💰 Token usage comparison for different storage formats:")
    print("-" * 70)

    for scenario_name, content in scenarios:
        _, _, token_count = analyze_tokenization(content)
        chars = len(content)
        efficiency = chars / token_count

        print(
            f"{scenario_name:20} | {chars:4d} chars | {token_count:3d} tokens | {efficiency:.1f} chars/token"
        )


def main():
    """Main function to run all demonstrations"""

    print("🎯 WALLET ADDRESS TOKENIZATION DEMO")
    print("This demonstrates how GPT tokenizes cryptocurrency wallet addresses")
    print("Understanding tokenization is crucial for efficient context usage!\n")

    # Run different analyses
    try:
        # 1. Compare different address formats
        compare_address_formats()

        # 2. Token cost analysis
        token_cost_analysis()

        # 3. Interactive mode
        print(f"\n{'='*70}")
        choice = input("Want to try the interactive tokenizer? (y/n): ").lower().strip()
        if choice == "y":
            interactive_tokenizer()

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Install required packages:")
        print("   pip install tiktoken")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
