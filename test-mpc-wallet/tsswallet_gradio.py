import gradio as gr
import os
from web3 import Web3
from eth_account import Account
from dotenv import load_dotenv
from tsswallet import (
    setup_tss_key,
    send_eth_with_tss_participants,
    TSSKeyData,
    EthereumTSS,
)
import json

# Global variables to store TSS data
global_key_data = None


def init_tss(private_key: str) -> str:
    """Initialize TSS with the given private key"""
    try:
        # Add warning about private key usage
        if not private_key:
            return "Please provide a private key. WARNING: This is a demo interface - never use real private keys!"

        # Remove 0x prefix if present
        if private_key.startswith("0x"):
            private_key = private_key[2:]

        global global_key_data
        # Setup TSS with threshold 2 and 3 parties
        global_key_data = setup_tss_key(private_key, threshold=2, num_parties=3)

        # Return success message with TSS address
        return (
            f"TSS initialized successfully!\nTSS Address: {global_key_data.tss_address}"
        )
    except Exception as e:
        return f"Error initializing TSS: {str(e)}"


def verify_tss() -> str:
    """Verify TSS setup by generating partial signatures"""
    try:
        if global_key_data is None:
            return "Please initialize TSS first!"

        # Create a dummy message to verify
        dummy_msg = Web3.keccak(text="verify")
        common_seed = Web3.keccak(text="test_seed")

        # Initialize TSS
        eth_tss = EthereumTSS()

        # Generate partial signatures for verification
        verification_result = []
        for i in range(global_key_data.threshold):
            partial_sig = eth_tss.create_partial_signature(
                global_key_data.parties[i], dummy_msg, common_seed
            )
            r, s_bytes, k, R_bytes = partial_sig
            verification_result.append(
                f"Party {i+1} signature verified - r: 0x{hex(r)[2:]}..."
            )

        return "\n".join(["Verification successful!"] + verification_result)
    except Exception as e:
        return f"Error during verification: {str(e)}"


def send_transaction(to_address: str, amount: float) -> str:
    """Send a transaction using TSS"""
    try:
        if global_key_data is None:
            return "Please initialize TSS first!"

        if not Web3.is_address(to_address):
            return "Invalid Ethereum address!"

        if amount <= 0:
            return "Amount must be greater than 0!"

        # Connect to Cronos testnet
        w3 = Web3(Web3.HTTPProvider("https://evm-t3.cronos.org/"))
        if not w3.is_connected():
            return "Failed to connect to the network!"

        # Get initial balances
        sender_balance_before = w3.eth.get_balance(global_key_data.tss_address)
        receiver_balance_before = w3.eth.get_balance(to_address)

        result_lines = [
            "=== Initial Balances ===",
            f"Sender balance: {w3.from_wei(sender_balance_before, 'ether')} TCRO",
            f"Receiver balance: {w3.from_wei(receiver_balance_before, 'ether')} TCRO",
        ]

        # Create a list to capture all output
        all_output = []

        def capture_print(text):
            all_output.append(str(text))

        # Monkey patch the print function in the module
        import builtins

        original_print = builtins.print
        builtins.print = capture_print

        try:
            # Send transaction using TSS
            result = send_eth_with_tss_participants(global_key_data, to_address, amount)
        finally:
            # Restore original print
            builtins.print = original_print

        # Add captured output
        result_lines.extend(all_output)

        return "\n".join(result_lines)
    except Exception as e:
        return f"Error sending transaction: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="TSS Wallet") as app:
    gr.Markdown("# Threshold Signature Scheme (TSS) Wallet")
    gr.Markdown(
        """
    ## ⚠️ Warning
    This is a demonstration interface for TSS wallet functionality. 
    - Never use real private keys in this interface
    - This is for testing and educational purposes only
    - Use testnet TCRO only
    """
    )

    gr.Markdown("## Initialize TSS")

    with gr.Row():
        private_key_input = gr.Textbox(
            label="Private Key",
            placeholder="Enter your private key (with or without 0x prefix)",
            type="password",
        )
        init_button = gr.Button("Initialize TSS")

    init_output = gr.Textbox(label="Initialization Result", lines=3)

    gr.Markdown("## Verify TSS Setup")
    verify_button = gr.Button("Verify TSS Setup")
    verify_output = gr.Textbox(label="Verification Result", lines=5)

    gr.Markdown("## Send Transaction")
    with gr.Row():
        to_address_input = gr.Textbox(
            label="To Address", placeholder="Enter recipient's Ethereum address"
        )
        amount_input = gr.Number(label="Amount (TCRO)", value=0.1)

    send_button = gr.Button("Send Transaction")
    send_output = gr.Textbox(label="Transaction Result", lines=30, max_lines=50)

    # Connect components
    init_button.click(init_tss, inputs=[private_key_input], outputs=[init_output])

    verify_button.click(verify_tss, inputs=[], outputs=[verify_output])

    send_button.click(
        send_transaction, inputs=[to_address_input, amount_input], outputs=[send_output]
    )

if __name__ == "__main__":
    app.launch()
