"""
Show how LayerNorm parameters γ and β are learned during training
"""

import torch
import torch.nn as nn
import torch.optim as optim


def show_layernorm_learning():
    """Demonstrate that γ and β are trainable parameters"""
    print("=" * 50)
    print("🎓 LAYERNORM PARAMETERS ARE LEARNED!")
    print("=" * 50)

    # Create a LayerNorm layer
    d_model = 4
    layer_norm = nn.LayerNorm(d_model)

    print(f"🔧 LayerNorm layer created with d_model = {d_model}")

    # Show initial parameters
    print(f"\n📊 Initial parameters:")
    print(f"   γ (weight): {layer_norm.weight}")
    print(f"   β (bias):   {layer_norm.bias}")
    print(f"   • γ initialized to all 1s (vector with {d_model} elements)")
    print(f"   • β initialized to all 0s (vector with {d_model} elements)")
    print(f"   • NOT scalars! Each dimension has its own γ and β")

    print(f"\n🔢 Parameter shapes:")
    print(f"   γ shape: {layer_norm.weight.shape}")
    print(f"   β shape: {layer_norm.bias.shape}")
    print(
        f"   Total learnable parameters: {layer_norm.weight.numel() + layer_norm.bias.numel()}"
    )

    # Check if they require gradients
    print(f"\n🎯 Are they trainable?")
    print(f"   γ requires_grad: {layer_norm.weight.requires_grad}")
    print(f"   β requires_grad: {layer_norm.bias.requires_grad}")

    # Show they have gradients after backward pass
    input_data = torch.randn(2, 4)  # batch_size=2, d_model=4
    output = layer_norm(input_data)
    loss = output.sum()  # Dummy loss
    loss.backward()

    print(f"\n🔄 After backward pass:")
    print(f"   γ gradients: {layer_norm.weight.grad}")
    print(f"   β gradients: {layer_norm.bias.grad}")
    print(f"   • Each dimension gets its own gradient!")
    print(f"   • Gradients are computed - they can be updated!")


def simulate_training():
    """Simulate how parameters change during training"""
    print(f"\n" + "=" * 50)
    print("🏃 SIMULATE TRAINING - WATCH PARAMETERS CHANGE")
    print("=" * 50)

    # Simple setup
    d_model = 3
    layer_norm = nn.LayerNorm(d_model)
    optimizer = optim.SGD(layer_norm.parameters(), lr=0.1)

    print(f"📊 Training setup:")
    print(f"   d_model: {d_model}")
    print(f"   Learning rate: 0.1")
    print(f"   Optimizer: SGD")

    # Show parameter evolution
    print(f"\n📈 Parameter evolution during training:")

    for epoch in range(5):
        # Forward pass
        input_data = torch.randn(1, d_model)
        output = layer_norm(input_data)

        # Dummy loss (just sum of outputs)
        loss = output.sum()

        print(f"\n   Epoch {epoch}:")
        print(f"   γ: {layer_norm.weight.data}")
        print(f"   β: {layer_norm.bias.data}")

        # Backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"\n✅ Notice how γ and β changed from their initial values!")
    print(f"   • They learned to scale and shift the normalized values")
    print(f"   • This helps the model adapt to the specific data distribution")


def scalar_vs_vector_demo():
    """Show clearly which are scalars vs vectors"""
    print(f"\n" + "=" * 50)
    print("📏 SCALAR vs VECTOR IN LAYER NORMALIZATION")
    print("=" * 50)

    # Example with d_model = 4
    x = torch.tensor([10.0, 2.0, 30.0, 5.0])
    d_model = len(x)

    print(f"📊 Input vector: {x}")
    print(f"   d_model = {d_model}")

    # Calculate mean and std (scalars)
    mu = x.mean()
    sigma = x.std()

    print(f"\n🔢 Statistics (SCALARS - single numbers):")
    print(f"   μ (mean) = sum({x}) / {d_model} = {mu:.2f}")
    print(f"   σ (std) = {sigma:.2f}")
    print(f"   • μ and σ are computed from ALL dimensions → scalars")

    # Normalize
    normalized = (x - mu) / sigma

    print(f"\n✨ After normalization: {normalized}")
    print(f"   All elements normalized using the SAME μ and σ")

    # Layer norm parameters (vectors)
    gamma = torch.tensor([2.0, 0.5, 1.0, 3.0])  # One per dimension
    beta = torch.tensor([0.1, -0.2, 0.0, 0.5])  # One per dimension

    print(f"\n🎯 Learnable parameters (VECTORS - one per dimension):")
    print(f"   γ = {gamma} (shape: {gamma.shape})")
    print(f"   β = {beta} (shape: {beta.shape})")
    print(f"   • Each dimension gets its own γ and β")

    # Final result
    final = gamma * normalized + beta

    print(f"\n📈 Element-wise calculation:")
    for i in range(d_model):
        print(
            f"   dim {i}: {gamma[i]:.1f} * {normalized[i]:.2f} + {beta[i]:.1f} = {final[i]:.2f}"
        )

    print(f"\n✅ Final output: {final}")

    print(f"\n💡 Summary:")
    print(f"   • μ, σ: SCALARS (1 number each)")
    print(f"   • γ, β: VECTORS ({d_model} numbers each)")
    print(f"   • Total learnable params: {len(gamma) + len(beta)}")


def why_learnable_parameters():
    """Explain why γ and β need to be learnable"""
    print(f"\n" + "=" * 50)
    print("❓ WHY ARE γ AND β LEARNABLE?")
    print("=" * 50)

    print(f"\n🤔 Without learnable parameters:")
    print(f"   • Normalization always forces mean=0, std=1")
    print(f"   • But maybe the model needs different scales per dimension")
    print(f"   • The model has no way to adjust this!")

    print(f"\n✅ With learnable γ and β:")
    print(f"   • Model can learn optimal scale (γ) and shift (β) per dimension")
    print(f"   • Example: γ=[2.0, 0.5, 1.5], β=[0.1, -0.3, 0.8]")
    print(f"   • Each feature dimension can have its own normalization!")

    print(f"\n🧠 Why vectors not scalars?")
    print(f"   • Different dimensions represent different features:")
    print(f"     - Dim 0: verb tense information")
    print(f"     - Dim 1: subject-object relationships")
    print(f"     - Dim 2: sentiment polarity")
    print(f"     - Dim 3: grammatical number")
    print(f"   • Each feature needs its own optimal scale/shift!")

    # Example showing the effect
    d_model = 3
    x = torch.tensor([1.0, 2.0, 3.0])

    # Standard normalization (mean=0, std=1)
    normalized = (x - x.mean()) / x.std()

    # With learned parameters
    gamma = torch.tensor([2.0, 0.5, 1.5])  # Learned scales
    beta = torch.tensor([0.1, -0.3, 0.8])  # Learned shifts
    final = gamma * normalized + beta

    print(f"\n📊 Example:")
    print(f"   Input: {x}")
    print(f"   After normalization: {normalized}")
    print(f"   With γ={gamma}, β={beta}:")
    print(f"   Final output: {final}")
    print(f"   → Model learned different transforms per dimension!")


def main():
    show_layernorm_learning()
    simulate_training()
    scalar_vs_vector_demo()
    why_learnable_parameters()

    print(f"\n" + "=" * 50)
    print(f"🎓 KEY INSIGHTS")
    print(f"=" * 50)
    print(f"1. γ and β are trainable parameters (requires_grad=True)")
    print(f"2. They start at γ=1, β=0 (identity transformation)")
    print(f"3. During training, they learn optimal scale/shift values")
    print(f"4. This allows the model to adapt normalization to its needs")
    print(f"5. Without them, normalization would be too restrictive")
    print(f"6. They're updated by the optimizer just like other weights!")


if __name__ == "__main__":
    main()
