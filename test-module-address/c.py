import hashlib
import bech32    

# read hex string to byte array


def get_module_account(h,prefix):    
    buf =  bytes.fromhex(h)
    print("buf=", buf.hex(), "  buf length=", len(buf))
    #computed= hashlib.sha256(buf).digest()[:20]
    #print("sha256: ", hashlib.sha256(buf).digest().hex())
    #print("computed: ", computed.hex())
    #ret=bech32.encode(prefix, 0,computed)
    ret=bech32.encode(prefix,0,buf)
    return ret
    

# read string from stdin
print("Enter hex string:")
h=input()
print("Enter bech32 prefix:")
prefix=input()
addr= get_module_account(h,prefix)
print(addr)