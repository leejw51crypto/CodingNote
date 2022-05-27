use num_bigint::{BigInt, ToBigInt};
use rand::Rng;
use sha2::Digest;
fn main() {
    // Step 1: Generate the secret values a and b
    let mut rng = rand::thread_rng();
    let a = rng.gen_range(1..100).to_bigint().unwrap();
    let b = rng.gen_range(1..100).to_bigint().unwrap();

    // Step 2: Generate the public parameters
    let p = BigInt::from(104723);
    let g = BigInt::from(3);

    // Step 3: Generate the random value r
    let r = rng.gen_range(1..104723).to_bigint().unwrap();

    // Step 4: Calculate A = g^a * r^p mod p and B = g^b * r^p mod p
    let A = g.modpow(&a, &p) * r.modpow(&p, &p) % &p;
    let B = g.modpow(&b, &p) * r.modpow(&p, &p) % &p;

    // Step 5: Generate the challenge value e
    let mut hasher = sha2::Sha256::new();
    hasher.update(A.to_string().as_bytes());
    hasher.update(B.to_string().as_bytes());
    let e = BigInt::from_bytes_be(num_bigint::Sign::Plus, &hasher.finalize()[..]);

    //let z = (r - &e * &(a + &b)) % &p;
    // Step 6: Calculate z = r - e*(a+b) mod p
    //let z = r.mod_sub(&(&e * &(a + &b)), &p);
    let mut z = (r - &e * &(a + &b)) % &p;
    if z.sign() == num_bigint::Sign::Minus {
        z += &p;
    }

    // Step 7: Verify the proof
    let A_prime = g.modpow(&z, &p) * A.modpow(&e, &p) % &p;
    let B_prime = g.modpow(&z, &p) * B.modpow(&e, &p) % &p;
    let mut hasher = sha2::Sha256::new();
    hasher.update(A_prime.to_string().as_bytes());
    hasher.update(B_prime.to_string().as_bytes());
    let e_prime = BigInt::from_bytes_be(num_bigint::Sign::Plus, &hasher.finalize()[..]);
    assert_eq!(e, e_prime);
}
