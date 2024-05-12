import os
import json
from eth_account import Account
from eth_utils import to_checksum_address
from eth_account.messages import encode_defunct
from datetime import datetime, timedelta

Account.enable_unaudited_hdwallet_features()

def get_account_from_mnemonic(mnemonic_phrase):
    """
    Retrieves an Ethereum account from a mnemonic phrase.
    Returns the account, private key, and checksum address.
    """
    account = Account.from_mnemonic(mnemonic_phrase, account_path="m/44'/60'/0'/0/0")
    private_key = account._private_key.hex()
    address = to_checksum_address(account.address)
    print(f"Address: {address}")
    return account, private_key, address

def create_login_message(address):
    """
    Creates a login message with the given wallet address and current timestamps.
    Returns the login message as a dictionary.
    """
    utc_now = datetime.utcnow()
    local_now = datetime.now()
    message = {
        "action": "login",
        "username": "player123",
        "game_id": "game456",
        "wallet_address": address,
        "timestamp": utc_now.isoformat() + "Z",
        "localtimestamp": local_now.isoformat(),
        "expiration_utc": (utc_now + timedelta(days=1)).isoformat() + "Z",
        "expiration_local": (local_now + timedelta(days=1)).isoformat()
    }
    return message

def sign_message(message, private_key):
    """
    Signs a message with the given private key.
    Returns the message as JSON, the message hash, and the signed message.
    """
    message_json = json.dumps(message, indent=4)
    message_hash = encode_defunct(text=message_json)
    signed_message = Account.sign_message(message_hash, private_key=private_key)
    return message_json, message_hash, signed_message

def verify_signature(message_hash, signed_message, address):
    """
    Verifies the signature of a signed message against the given message hash and address.
    Returns True if the signature is valid, False otherwise.
    """
    recovered_address = Account.recover_message(message_hash, signature=signed_message.signature)
    return recovered_address == address

def test_valid_signature():
    """
    Test case for a valid signature.
    """
    mnemonic_phrase = os.environ.get("MY_MNEMONICS")
    if mnemonic_phrase is None:
        raise ValueError("MY_MNEMONICS environment variable is not set")

    account, private_key, address = get_account_from_mnemonic(mnemonic_phrase)
    message = create_login_message(address)
    message_json, message_hash, signed_message = sign_message(message, private_key)
    is_valid = verify_signature(message_hash, signed_message, address)
    print(f"\nSignature is valid: {is_valid}")

def test_invalid_signature():
    """
    Test case for an invalid signature.
    """
    mnemonic_phrase = os.environ.get("MY_MNEMONICS")
    if mnemonic_phrase is None:
        raise ValueError("MY_MNEMONICS environment variable is not set")

    account, private_key, address = get_account_from_mnemonic(mnemonic_phrase)
    message = create_login_message(address)
    message_json, message_hash, signed_message = sign_message(message, private_key)

    # Modify the timestamp to invalidate the signature
    modified_message = json.loads(message_json)
    modified_message["timestamp"] = (datetime.utcnow() + timedelta(hours=1)).isoformat() + "Z"
    modified_message_json = json.dumps(modified_message, indent=4)
    modified_message_hash = encode_defunct(text=modified_message_json)

    is_valid = verify_signature(modified_message_hash, signed_message, address)
    print(f"\nSignature is valid: {is_valid}")

if __name__ == "__main__":
    print("Test Case 1: Valid Signature")
    test_valid_signature()

    print("\nTest Case 2: Invalid Signature")
    test_invalid_signature()
