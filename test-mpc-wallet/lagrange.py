"""
Lagrange Interpolation for Secret Sharing

Mathematical Background:
------------------------
Shamir's Secret Sharing splits a secret S into n shares, where any t shares can reconstruct S.
The scheme is based on polynomial interpolation.

Key Concepts:
1. Polynomial Creation:
   - Create a random polynomial f(x) of degree t-1:
   f(x) = a₀ + a₁x + a₂x² + ... + aₜ₋₁xᵗ⁻¹
   where:
   - a₀ = S (the secret)
   - a₁, a₂, ..., aₜ₋₁ are random coefficients

   Example for t=3 (2nd degree polynomial):
   f(x) = S + 7x + 3x²   (where 7,3 are random)
   
   Visual representation of shares:
   y ↑
     |    • P₃(3,f(3))
     |       • P₂(2,f(2))
     |   • P₁(1,f(1))
     | • S(0,f(0))
     +------------→ x

2. Share Generation:
   - Each participant i gets a point (i, f(i))
   - Share for participant i: sᵢ = f(i)
   
   Example (t=3,n=5):
   - 5 shares generated: (1,f(1)), (2,f(2)), ..., (5,f(5))
   - Any 3 shares can reconstruct S
   - 2 or fewer shares reveal nothing about S

3. Secret Reconstruction:
   Given t shares (x₁,y₁), (x₂,y₂), ..., (xₜ,yₜ), we can reconstruct f(x):

   f(x) = ∑(i=1 to t) yᵢ · ℓᵢ(x)

   where ℓᵢ(x) are Lagrange basis polynomials:
   ℓᵢ(x) = ∏(j≠i) (x-xⱼ)/(xᵢ-xⱼ)

4. Secret Recovery:
   The secret S is recovered by evaluating f(0):
   S = f(0) = ∑(i=1 to t) yᵢ · λᵢ

   where λᵢ (Lagrange coefficients) = ℓᵢ(0):
   λᵢ = ∏(j≠i) (-xⱼ)/(xᵢ-xⱼ)

Security Properties:
-------------------
1. Perfect Security: With t-1 or fewer shares, no information about S is revealed
   Example: With t=3, even having 2 points:
   - Infinite polynomials of degree 2 pass through 2 points
   - Each polynomial has a different y-intercept (secret)
   - All possible secrets are equally likely

2. Information Theoretic: Security doesn't rely on computational hardness
3. Minimal: Each share must be at least as large as the secret

All calculations are performed modulo a prime p (in our case, the secp256k1 curve order)
to ensure security and proper field arithmetic.
"""


def mod_inverse(a: int, m: int) -> int:
    """Calculate modular multiplicative inverse"""

    def extended_gcd(a: int, b: int):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y

    gcd, x, _ = extended_gcd(a, m)
    if gcd != 1:
        raise ValueError("Modular inverse does not exist")
    return (x % m + m) % m


def lagrange_coefficient(parties: list, j: int, x: int = 0) -> int:
    """Calculate Lagrange coefficient λⱼ for party j

    λⱼ = ∏(m≠j) (x-m)/(j-m)
    When x=0: λⱼ = ∏(m≠j) (-m)/(j-m)

    All calculations are done modulo the curve order.
    """
    curve_order = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

    num, den = 1, 1
    for m in parties:
        if m != j:
            # Calculate numerator: (x-m) mod curve_order
            num = (num * (x - m)) % curve_order
            # Calculate denominator: (j-m) mod curve_order
            den = (den * (j - m)) % curve_order
    # Final coefficient = num * den^(-1) mod curve_order
    return (num * mod_inverse(den, curve_order)) % curve_order


def main():
    """
    Demonstrate 2-out-of-3 Shamir's Secret Sharing

    Using polynomial f(x) = secret + 7x
    - f(0) = secret (this is what we want to recover)
    - f(1) = secret + 7 (share for party 1)
    - f(2) = secret + 14 (share for party 2)
    - f(3) = secret + 21 (share for party 3)
    """
    # Load environment variables
    from dotenv import load_dotenv
    import os

    load_dotenv()

    # Get secret from environment variables
    secret = int(os.getenv("TEST_SECRET_NUMBER"))
    if secret is None:
        raise ValueError("TEST_SECRET_NUMBER not found in .env file")

    # Coefficient for the linear term (degree 1)
    coefficient = 7

    # Generate shares: f(i) for i = 1,2,3
    shares = {}
    for i in range(1, 4):  # Party IDs: 1, 2, 3
        # Calculate f(i) = secret + coefficient * i
        share = (
            secret + coefficient * i
        ) % 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        shares[i] = share

    print("Original secret (s):", secret)
    print("\nShares (sᵢ = f(i)):")
    for party_id, share in shares.items():
        print(f"Party {party_id} (s_{party_id}): {share}")

    # Reconstruct using parties 1 and 2
    participating_parties = [1, 2]
    print("\nReconstructing with parties:", participating_parties)

    # s = ∑(i∈parties) sᵢ · λᵢ
    reconstructed_secret = 0
    for party_id in participating_parties:
        # Calculate λᵢ for this party
        coeff = lagrange_coefficient(participating_parties, party_id)
        print(f"\nLagrange coefficient λ_{party_id}: {coeff}")

        # Add sᵢ · λᵢ to the sum
        reconstructed_secret = (
            reconstructed_secret + shares[party_id] * coeff
        ) % 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

    print("\nReconstructed secret (s):", reconstructed_secret)
    print("Reconstruction successful:", secret == reconstructed_secret)

    # Try another combination: parties 2 and 3
    participating_parties = [2, 3]
    print("\nReconstructing with parties:", participating_parties)

    reconstructed_secret = 0
    for party_id in participating_parties:
        coeff = lagrange_coefficient(participating_parties, party_id)
        print(f"\nLagrange coefficient λ_{party_id}: {coeff}")
        reconstructed_secret = (
            reconstructed_secret + shares[party_id] * coeff
        ) % 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

    print("\nReconstructed secret (s):", reconstructed_secret)
    print("Reconstruction successful:", secret == reconstructed_secret)


if __name__ == "__main__":
    main()
