import hashlib
import bech32    


def get_module_account(modulename,prefix):    
    buf = modulename.encode()
    print("hex=", buf.hex(), "  buf length=", len(buf))
    computed= hashlib.sha256(buf).digest()[:20]
    print("sha256: ", hashlib.sha256(buf).digest().hex())
    print("computed: ", computed.hex())
    ret=bech32.bech32_encode(prefix, bech32.convertbits(computed, 8, 5))
    return ret
    

# read string from stdin
print("Enter modulename:")
modulename=input()
print("Enter bech32 prefix:")
prefix=input()
addr= get_module_account(modulename,prefix)
print(addr)