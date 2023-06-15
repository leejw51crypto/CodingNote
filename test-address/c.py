## read the json from file my.json
import json
import bech32

def convert(bech32_address) :
    _, bz = bech32.bech32_decode(bech32_address)
    hexbytes=bytes(bech32.convertbits(bz, 5, 8))
    eth_address = '0x' + hexbytes.hex()
    return eth_address


with open('my.json') as f:
    data = json.load(f)
    # data is list, iterate
    for item in data:
        # get address from item
        address = item['address']
        address_hex= convert(address)
        item['address_hex'] = address_hex

# write data to my2.json
with open('my2.json', 'w') as f:
    json.dump(data, f)


