"""
Softmax Example for Understanding Attention
Shows how softmax converts raw scores to attention weights
"""

import numpy as np
import torch
import torch.nn.functional as F


def demo_basic_softmax():
    """Basic softmax demonstration"""
    print("=" * 60)
    print("🧮 BASIC SOFTMAX DEMONSTRATION")
    print("=" * 60)

    # Raw attention scores (before softmax)
    scores = torch.tensor([2.0, 1.0, 0.5])
    print(f"\n📊 Raw attention scores: {scores}")
    print(
        f"   These could be: [similarity to 'I', similarity to 'love', similarity to 'you']"
    )

    # Apply softmax
    weights = F.softmax(scores, dim=0)
    print(f"\n✨ After softmax: {weights}")
    print(f"   Sum of weights: {weights.sum():.3f} (always equals 1.0)")

    print("\n📈 What softmax does:")
    print(
        f"   1. Exponentiates: e^2.0={torch.exp(scores[0]):.2f}, e^1.0={torch.exp(scores[1]):.2f}, e^0.5={torch.exp(scores[2]):.2f}"
    )
    print(f"   2. Normalizes: Divides each by sum ({torch.exp(scores).sum():.2f})")
    print(f"   3. Result: Higher scores get more weight, but all sum to 1.0")


def demo_attention_example():
    """Real attention example with English-Italian translation"""
    print("\n" + "=" * 60)
    print("🌍 CROSS-ATTENTION EXAMPLE: Generating 'amo' (love)")
    print("=" * 60)

    # English words (encoder output)
    english_words = ["I", "love", "you"]

    # Raw attention scores when decoder generates 'amo'
    # High score for 'love' because 'amo' means 'love'
    raw_scores = torch.tensor([0.5, 3.0, 0.2])

    print(f"\n🔤 English words: {english_words}")
    print(f"🎯 Generating Italian word: 'amo' (love)")
    print(f"\n📊 Raw attention scores (Q·K^T):")
    for i, (word, score) in enumerate(zip(english_words, raw_scores)):
        print(f"   {word:>6}: {score:5.1f}")

    # Apply softmax
    attention_weights = F.softmax(raw_scores, dim=0)

    print(f"\n✨ Attention weights (after softmax):")
    for i, (word, weight) in enumerate(zip(english_words, attention_weights)):
        bar_length = int(weight * 50)  # Scale for visualization
        bar = "█" * bar_length
        print(f"   {word:>6}: {weight:.3f} {bar}")

    print(f"\n💡 Interpretation:")
    print(f"   • 'amo' pays {attention_weights[1]:.1%} attention to 'love'")
    print(f"   • 'amo' pays {attention_weights[0]:.1%} attention to 'I'")
    print(f"   • 'amo' pays {attention_weights[2]:.1%} attention to 'you'")
    print(f"   • This makes sense: 'amo' = 'love' in Italian!")


def demo_temperature_effect():
    """Show how temperature affects softmax sharpness"""
    print("\n" + "=" * 60)
    print("🌡️ TEMPERATURE EFFECT ON SOFTMAX")
    print("=" * 60)

    scores = torch.tensor([2.0, 1.5, 0.5])
    english_words = ["I", "love", "you"]

    temperatures = [0.1, 1.0, 2.0, 10.0]

    print(f"\n📊 Original scores: {scores}")
    print(f"🔤 Words: {english_words}")

    for temp in temperatures:
        # Apply temperature scaling before softmax
        scaled_scores = scores / temp
        weights = F.softmax(scaled_scores, dim=0)

        print(f"\n🌡️ Temperature = {temp}")
        print(f"   Scaled scores: {scaled_scores}")
        print(f"   Attention weights:")
        for word, weight in zip(english_words, weights):
            bar = "█" * int(weight * 30)
            print(f"     {word:>6}: {weight:.3f} {bar}")

    print(f"\n💡 Temperature Effects:")
    print(f"   • Low temp (0.1):  Sharp, peaked attention (confident)")
    print(f"   • Medium temp (1.0): Balanced attention")
    print(f"   • High temp (10.0): Flat, uniform attention (uncertain)")


def demo_attention_matrix():
    """Show full attention matrix for sequence"""
    print("\n" + "=" * 60)
    print("📋 FULL ATTENTION MATRIX EXAMPLE")
    print("=" * 60)

    english = ["I", "love", "you"]
    italian = ["ti", "amo"]

    # Raw scores matrix (decoder positions × encoder positions)
    # Each row is one decoder position attending to all encoder positions
    raw_scores = torch.tensor(
        [
            [0.2, 0.3, 2.8],  # When generating 'ti', attend most to 'you'
            [0.1, 2.9, 0.5],  # When generating 'amo', attend most to 'love'
        ]
    )

    print(f"🔤 English (encoder): {english}")
    print(f"🔤 Italian (decoder): {italian}")

    print(f"\n📊 Raw attention scores:")
    print("     " + "".join([f"{word:>8}" for word in english]))
    for i, (ital_word, scores_row) in enumerate(zip(italian, raw_scores)):
        print(f"{ital_word:>4} " + "".join([f"{score:8.1f}" for score in scores_row]))

    # Apply softmax to each row (each decoder position)
    attention_matrix = F.softmax(raw_scores, dim=1)

    print(f"\n✨ Attention matrix (after row-wise softmax):")
    print("     " + "".join([f"{word:>8}" for word in english]))
    for i, (ital_word, weights_row) in enumerate(zip(italian, attention_matrix)):
        print(
            f"{ital_word:>4} " + "".join([f"{weight:8.3f}" for weight in weights_row])
        )

    print(f"\n🎯 Translation alignments learned:")
    for i, ital_word in enumerate(italian):
        max_idx = torch.argmax(attention_matrix[i])
        max_weight = attention_matrix[i][max_idx]
        eng_word = english[max_idx]
        print(f"   '{ital_word}' ← {max_weight:.1%} attention ← '{eng_word}'")


def demo_why_softmax():
    """Explain why we use softmax instead of alternatives"""
    print("\n" + "=" * 60)
    print("❓ WHY SOFTMAX? COMPARISON WITH ALTERNATIVES")
    print("=" * 60)

    scores = torch.tensor([2.0, 1.0, 0.1])
    words = ["love", "like", "hate"]

    print(f"📊 Raw scores: {scores}")
    print(f"🔤 Words: {words}")

    # Softmax (what we use)
    softmax_weights = F.softmax(scores, dim=0)

    # Alternative 1: Just normalize by sum (linear)
    linear_weights = scores / scores.sum()

    # Alternative 2: Max only (hard attention)
    max_weights = torch.zeros_like(scores)
    max_weights[torch.argmax(scores)] = 1.0

    print(f"\n📈 Comparison:")
    print(f"{'Method':<12} {'love':<8} {'like':<8} {'hate':<8} {'Properties'}")
    print(f"{'-'*60}")
    print(
        f"{'Softmax':<12} {softmax_weights[0]:.3f}    {softmax_weights[1]:.3f}    {softmax_weights[2]:.3f}    Smooth, differentiable"
    )
    print(
        f"{'Linear':<12} {linear_weights[0]:.3f}    {linear_weights[1]:.3f}    {linear_weights[2]:.3f}    Can be negative!"
    )
    print(
        f"{'Max-only':<12} {max_weights[0]:.3f}    {max_weights[1]:.3f}    {max_weights[2]:.3f}    Not differentiable"
    )

    print(f"\n✅ Why softmax wins:")
    print(f"   • Always positive (probabilities)")
    print(f"   • Sums to 1.0 (probability distribution)")
    print(f"   • Differentiable (can train with gradients)")
    print(f"   • Emphasizes larger values while keeping smaller ones")
    print(f"   • Smooth (small changes in input → small changes in output)")


def main():
    print("🎓 SOFTMAX IN ATTENTION MECHANISMS")
    print("Understanding how raw scores become attention weights\n")

    demo_basic_softmax()
    demo_attention_example()
    demo_temperature_effect()
    demo_attention_matrix()
    demo_why_softmax()

    print("\n" + "=" * 60)
    print("🎯 KEY TAKEAWAYS")
    print("=" * 60)
    print("1. Softmax converts raw scores to probabilities (0-1, sum=1)")
    print("2. Higher scores get exponentially more weight")
    print("3. All positions get some attention (soft vs hard attention)")
    print("4. Temperature controls how 'sharp' the attention is")
    print("5. In cross-attention: decoder queries 'compete' for encoder keys")
    print("6. The resulting weights determine how much of each encoder")
    print("   value to include in the final representation")


if __name__ == "__main__":
    main()
