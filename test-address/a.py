import bech32

# read eth_address from console
eth_address = input("Please enter a ethaddress: ")
eth_address_bytes = bytes.fromhex(eth_address[2:])
print("eth_address length: ", len(eth_address_bytes))
print("eth_address: ", eth_address_bytes.hex())
bz = bech32.convertbits(eth_address_bytes, 8, 5)
bech32_address = bech32.bech32_encode("crc",bz)
print(bech32_address)