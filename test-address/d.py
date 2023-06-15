import json
import bech32

# Function to convert bech32 address to hex string
def convert(bech32_address):
    _, bz = bech32.bech32_decode(bech32_address)
    hexbytes = bytes(bech32.convertbits(bz, 5, 8))
    eth_address = '0x' + hexbytes.hex()
    return eth_address

# Read data from my.json
with open('my.json') as f:
    data = json.load(f)

    # Iterate over the items in the list
    for item in data:
        # Get the bech32 address from the item
        address = item['address']

        # Convert bech32 address to hex string and add it to the item
        address_hex = convert(address)
        item['address_hex'] = address_hex

# Write updated data to my2.json
with open('my2.json', 'w') as f:
    json.dump(data, f)

