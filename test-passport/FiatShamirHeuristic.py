# Fiat-Shamir heuristic

# x= secret
# v= random()
# y= g^x ( mod p)
# t= g^v (mod p)
# c= H(g,y,t)
# r= v-cx
# y,t,r --> proof
# check t== g^r* y^c ?

# t= g^v  <- t is computed like this
# so  g^r * y^c = g^(v-cx)* g^(xc) = g^v


import hashlib
import random

def hash_function(*args):
    hash_object = hashlib.sha256()
    for arg in args:
        hash_object.update(str(arg).encode())
    return int(hash_object.hexdigest(), 16)

def generate_prime():
    # This is a placeholder. In practice, you should use a proper method to generate a large prime number.
    return 1000000007

def modular_pow(base, exponent, modulus):
    if modulus == 1:
        return 0
    result = 1
    base = base % modulus
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % modulus
        exponent = exponent >> 1
        base = (base * base) % modulus
    return result

def prove(g, p, x):
    y = modular_pow(g, x, p)
    v = random.randint(1, p-1)
    t = modular_pow(g, v, p)
    c = hash_function(g, y, t)
    r = (v - c * x) % (p-1)
    return y, t, r


def verify(g, p, y, t, r):
    c = hash_function(g, y, t)
    return t == (modular_pow(g, r, p) * modular_pow(y, c, p)) % p

# Example usage
p = generate_prime()
g = 2  # In practice, choose a suitable generator for the group
x = random.randint(1, p-1)  # Secret

y, t, r = prove(g, p, x)
assert verify(g, p, y, t, r), "Verification failed"
print("Verified Successfully")