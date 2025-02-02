import numpy as np
import matplotlib.pyplot as plt
from lagrange import lagrange_coefficient


def plot_secret_sharing():
    # Create figure and axis
    plt.figure(figsize=(12, 7))

    # Define the coefficients for f(x) = ax² + bx + secret
    secret = 10  # f(0) = 10
    a = 2  # coefficient of x²
    b = -3  # coefficient of x

    # Generate points for the polynomial f(x) = ax² + bx + secret
    x = np.linspace(-1, 4, 200)
    y = [a * xi**2 + b * xi + secret for xi in x]

    # Plot the polynomial
    plt.plot(x, y, "purple", label=f"f(x) = {a}x² + {b}x + {secret}", linewidth=2)

    # Plot the shares (points)
    shares = {}
    for i in range(1, 4):  # Party IDs: 1, 2, 3
        shares[i] = a * i**2 + b * i + secret
        plt.plot(i, shares[i], "o", color="red", markersize=10)
        plt.annotate(
            f"P{i}({i},{shares[i]})",
            (i, shares[i]),
            xytext=(10, 10),
            textcoords="offset points",
        )

    # Plot the secret point
    plt.plot(0, secret, "o", color="green", markersize=10)
    plt.annotate(
        f"Secret(0,{secret})", (0, secret), xytext=(10, 10), textcoords="offset points"
    )

    # Add grid and labels
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2-out-of-3 Secret Sharing\nf(x) = 2x² - 3x + 10")
    plt.legend()

    # Add horizontal and vertical lines at 0
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    plot_secret_sharing()
