"""
Layer Normalization Example - Used in Transformers
Shows how to normalize hidden states for stable training
"""

import torch
import torch.nn as nn


def layer_norm_example():
    """Simple example of layer normalization"""
    print("=" * 50)
    print("🔧 LAYER NORMALIZATION IN TRANSFORMERS")
    print("=" * 50)

    # Example: hidden states from one token
    # In real transformer: shape would be [batch_size, seq_len, d_model]
    # Here we show just one position: [d_model] = [4] for simplicity
    hidden_states = torch.tensor([10.0, 2.0, 30.0, 5.0])
    d_model = len(hidden_states)

    print(f"🧠 Hidden states (from attention/feed-forward): {hidden_states}")
    print(f"   Dimension: {d_model} (d_model)")

    print(f"\n🧮 Layer Normalization Formula:")
    print(f"   LayerNorm(x) = γ * (x - μ) / σ + β")
    print(f"   Where:")
    print(f"   • μ = mean of x")
    print(f"   • σ = standard deviation of x")
    print(f"   • γ = learned scale parameter (initialized to 1)")
    print(f"   • β = learned shift parameter (initialized to 0)")

    print(f"\n📈 Step-by-step calculation:")

    # Step 1: Calculate mean
    mean = hidden_states.mean()
    print(f"   1. Calculate mean (μ):")
    print(
        f"      μ = ({hidden_states[0]} + {hidden_states[1]} + {hidden_states[2]} + {hidden_states[3]}) / 4"
    )
    print(f"      μ = {mean:.2f}")

    # Step 2: Calculate variance and std
    variance = ((hidden_states - mean) ** 2).mean()
    std = torch.sqrt(variance + 1e-5)  # Add small epsilon for numerical stability

    print(f"\n   2. Calculate standard deviation (σ):")
    centered = hidden_states - mean
    print(f"      x - μ = {centered}")
    print(f"      (x - μ)² = {centered**2}")
    print(f"      variance = {variance:.2f}")
    print(f"      σ = √(variance + ε) = {std:.2f}")

    # Step 3: Normalize
    normalized = (hidden_states - mean) / std
    print(f"\n   3. Normalize:")
    print(f"      (x - μ) / σ = {normalized}")

    # Step 4: Scale and shift (γ and β)
    gamma = torch.ones(d_model)  # Learnable scale (initialized to 1)
    beta = torch.zeros(d_model)  # Learnable shift (initialized to 0)

    final_output = gamma * normalized + beta

    print(f"\n   4. Apply learned parameters:")
    print(f"      γ = {gamma} (learnable scale)")
    print(f"      β = {beta} (learnable shift)")
    print(f"      output = γ * normalized + β = {final_output}")

    # Verify with PyTorch LayerNorm
    layer_norm = nn.LayerNorm(d_model)
    torch_output = layer_norm(hidden_states)

    print(f"\n✅ PyTorch LayerNorm result: {torch_output}")
    print(f"   Mean of normalized output: {torch_output.mean():.6f} (≈ 0)")
    print(f"   Std of normalized output: {torch_output.std():.6f} (≈ 1)")

    print(f"\n🎯 Properties after layer norm:")
    print(f"   • Mean ≈ 0: {final_output.mean():.6f}")
    print(f"   • Standard deviation ≈ 1: {final_output.std():.6f}")
    print(f"   • Same shape as input: {final_output.shape}")


def why_layer_norm():
    """Explain why we use layer normalization"""
    print("\n" + "=" * 50)
    print("❓ WHY LAYER NORMALIZATION?")
    print("=" * 50)

    print("\n🎯 Problems without normalization:")

    # Example: unstable values
    unstable = torch.tensor([1000.0, 0.001, 500.0, 0.1])
    stable_after_norm = nn.LayerNorm(4)(unstable)

    print(f"   Before: {unstable}")
    print(f"   • Very different scales (0.001 vs 1000)")
    print(f"   • Hard for gradients to flow properly")
    print(f"   • Training becomes unstable")

    print(f"\n   After LayerNorm: {stable_after_norm}")
    print(f"   • All values have similar scale")
    print(f"   • Stable gradients for training")

    print(f"\n✅ Benefits of Layer Normalization:")
    print(f"   1. Faster training convergence")
    print(f"   2. More stable gradients")
    print(f"   3. Less sensitive to initialization")
    print(f"   4. Allows higher learning rates")
    print(f"   5. Reduces internal covariate shift")


def transformer_usage():
    """Show where LayerNorm is used in transformers"""
    print("\n" + "=" * 50)
    print("🏗️ LAYER NORM IN TRANSFORMER ARCHITECTURE")
    print("=" * 50)

    print("\n📋 Used in multiple places:")
    print("\n1️⃣ After Multi-Head Attention:")
    print("   x = LayerNorm(x + MultiHeadAttention(x))")
    print("   (Residual connection + normalization)")

    print("\n2️⃣ After Feed-Forward Network:")
    print("   x = LayerNorm(x + FeedForward(x))")
    print("   (Residual connection + normalization)")

    print("\n📊 Example flow in one transformer block:")

    # Simulate one transformer block
    d_model = 4
    x = torch.tensor([2.0, 1.0, 3.0, 0.5])  # Input

    print(f"\n   Input: {x}")

    # Simulate attention output (just add some values)
    attention_out = x + torch.tensor([0.1, -0.2, 0.3, -0.1])
    print(f"   After attention: {attention_out}")

    # Layer norm after attention
    norm1 = nn.LayerNorm(d_model)
    x_normed = norm1(attention_out)
    print(f"   After LayerNorm 1: {x_normed}")

    # Simulate feed-forward (just multiply by 2)
    ff_out = x_normed + (x_normed * 2)
    print(f"   After feed-forward: {ff_out}")

    # Layer norm after feed-forward
    norm2 = nn.LayerNorm(d_model)
    final_out = norm2(ff_out)
    print(f"   After LayerNorm 2: {final_out}")

    print(f"\n💡 Notice how values stay in reasonable range throughout!")


def main():
    layer_norm_example()
    why_layer_norm()
    transformer_usage()

    print(f"\n" + "=" * 50)
    print(f"🎓 SUMMARY")
    print(f"=" * 50)
    print(f"Layer Norm formula: LayerNorm(x) = γ * (x - μ) / σ + β")
    print(f"")
    print(f"What it does:")
    print(f"1. Normalizes each layer's output to mean=0, std=1")
    print(f"2. Applies learned scale (γ) and shift (β) parameters")
    print(f"3. Keeps values in stable range for training")
    print(f"4. Used after attention and feed-forward in transformers")
    print(f"5. Critical for deep network training stability")


if __name__ == "__main__":
    main()
