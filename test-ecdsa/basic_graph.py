"""
MIT License

Copyright (c) 2024 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

⚠️ EDUCATIONAL VISUALIZATION WARNING ⚠️
=====================================
This is a simplified visualization tool for educational purposes:
1. Shows only a small section of the secp256k1 curve
2. Uses floating-point arithmetic (not suitable for cryptographic operations)
3. Does not implement actual cryptographic operations
4. For visualization and learning purposes only

For cryptographic implementations:
- Use established cryptographic libraries
- Implement proper finite field arithmetic
- Follow cryptographic best practices
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_curve_section():
    """
    Plot a section of the secp256k1 curve: y² = x³ + 7
    Note: This plots a small, visible section of the curve for illustration.
    The actual curve extends much further and works in modular arithmetic.
    """
    # Create figure and axis
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Generate x values
    x = np.linspace(-5, 5, 1000)

    # Calculate y values (y = ±√(x³ + 7))
    # We need both positive and negative square roots
    y_pos = np.sqrt(x**3 + 7)
    y_neg = -np.sqrt(x**3 + 7)

    # Plot the curve
    plt.plot(x, y_pos, "b-", label="y² = x³ + 7")
    plt.plot(x, y_neg, "b-")

    # Add grid and labels
    plt.grid(True, alpha=0.3)
    plt.title("secp256k1 Curve Visualization\n(Small Section for Illustration)")
    plt.xlabel("x")
    plt.ylabel("y")

    # Set equal aspect ratio to maintain curve shape
    ax.set_aspect("equal")

    # Add legend
    plt.legend()

    # Add text explanation
    explanation = """
    secp256k1 Curve Properties:
    • Equation: y² = x³ + 7 (mod p)
    • Used in Bitcoin and Ethereum
    • Prime field modulus p = 2²⁵⁶ - 2³² - 977
    • Group order n ≈ 2²⁵⁶
    
    Note: This is a simplified visualization.
    The actual curve operates in finite field
    arithmetic modulo p, making it discrete
    rather than continuous.
    """
    plt.text(6, 0, explanation, fontsize=8, bbox=dict(facecolor="white", alpha=0.8))

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_curve_section()
