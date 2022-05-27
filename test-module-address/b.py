import bech32


    
def compute(a):
    b=bytearray.fromhex(a)
    ret=bech32.encode("bc", 0,b)
    return ret
    


print("enter hex string:")
a=input()
b=compute(a) 
print("result: ", b)
