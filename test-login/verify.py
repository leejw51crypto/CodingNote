from eth_account import Account
from eth_account.messages import encode_defunct

def verify_signature(message, r, s, v, expected_address):
    # Prepare the message hash
    message_hash = encode_defunct(text=message)

    # Convert r and s to bytes
    r_bytes = r.to_bytes(32, 'big')
    s_bytes = s.to_bytes(32, 'big')

    # Concatenate r_bytes, s_bytes, and v_bytes
    signature_bytes = r_bytes + s_bytes + v.to_bytes(1, 'big')

    # Recover the signer's address from the signature
    signer = Account.recover_message(message_hash, signature=signature_bytes)

    # Compare the recovered signer's address with the expected address
    return signer == expected_address

# Original message
message = "Hello Crypto"

# Read signature values from the user
r = int(input("Enter the value of r: "))
s = int(input("Enter the value of s: "))
v = int(input("Enter the value of v: "))

# Read expected signer's address from the user
expected_address = input("Enter the expected signer's address: ")

# Verify the signature
is_valid = verify_signature(message, r, s, v, expected_address)
print("Signature is valid:", is_valid)
