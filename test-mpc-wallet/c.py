import os
from dotenv import load_dotenv
from tsswallet import setup_tss_key, send_eth_with_tss_participants

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    to_address = os.getenv("MY_TO_ADDRESS")
    if not to_address:
        raise ValueError("MY_TO_ADDRESS not set in environment")

    private_key = os.getenv("MY_FULL_PRIVATEKEY")
    if not private_key:
        raise ValueError("MY_FULL_PRIVATEKEY not set in environment")

    # First phase: Split the private key (this is the only place where full private key is used)
    key_data = setup_tss_key(private_key)

    # Second phase: Use the split keys to send transaction (no access to full private key)
    send_eth_with_tss_participants(key_data, to_address, 0.1)
