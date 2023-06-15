import bech32

bech32_address = input("Please enter a crc address: ")
_, bz = bech32.bech32_decode(bech32_address)
hexbytes=bytes(bech32.convertbits(bz, 5, 8))
eth_address = '0x' + hexbytes.hex()
print(eth_address)