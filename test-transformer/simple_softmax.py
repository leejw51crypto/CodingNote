"""
Simple Softmax Example - Vocabulary Prediction
Shows how raw logits become word probabilities
"""

import torch
import torch.nn.functional as F


def vocab_softmax_example():
    """Simple example of logits -> probabilities for vocabulary"""
    print("=" * 50)
    print("📝 SOFTMAX FOR VOCABULARY PREDICTION")
    print("=" * 50)

    # Simple vocabulary
    vocab = ["cat", "dog", "bird", "fish"]
    vocab_size = len(vocab)

    # Raw logits from model (before softmax)
    # Higher logit = model thinks this word is more likely
    logits = torch.tensor([2.1, 3.5, 0.8, 1.2])

    print(f"🔤 Vocabulary: {vocab}")
    print(f"📊 Raw logits: {logits}")
    print("   (Higher logit = model thinks word is more likely)")

    print(f"\n🧮 Softmax Formula:")
    print(f"   softmax(x_i) = e^(x_i) / Σ(e^(x_j))")
    print(f"   Where x_i is logit for word i")

    print(f"\n📈 Step-by-step calculation:")

    # Step 1: Exponential
    exp_logits = torch.exp(logits)
    print(f"   1. Take exponential of each logit:")
    for i, (word, logit, exp_val) in enumerate(zip(vocab, logits, exp_logits)):
        print(f"      e^{logit:.1f} = {exp_val:.2f}  ({word})")

    # Step 2: Sum
    sum_exp = exp_logits.sum()
    print(f"\n   2. Sum all exponentials:")
    print(
        f"      {exp_logits[0]:.2f} + {exp_logits[1]:.2f} + {exp_logits[2]:.2f} + {exp_logits[3]:.2f} = {sum_exp:.2f}"
    )

    # Step 3: Normalize
    probabilities = exp_logits / sum_exp
    print(f"\n   3. Normalize (divide each by sum):")
    for i, (word, exp_val, prob) in enumerate(zip(vocab, exp_logits, probabilities)):
        print(f"      {exp_val:.2f} / {sum_exp:.2f} = {prob:.3f}  ({word})")

    # Use PyTorch softmax to verify
    torch_probs = F.softmax(logits, dim=0)
    print(f"\n✅ PyTorch softmax result: {torch_probs}")
    print(f"   Sum of probabilities: {torch_probs.sum():.3f} (always 1.0)")

    print(f"\n🎯 Interpretation:")
    for word, prob in zip(vocab, torch_probs):
        percentage = prob * 100
        bar = "█" * int(percentage / 5)  # Scale bar
        print(f"   {word:>4}: {prob:.3f} ({percentage:4.1f}%) {bar}")

    print(f"\n💡 Key insight:")
    print(f"   • Model is most confident about 'dog' ({torch_probs[1]:.1%})")
    print(f"   • Even low-scoring words get some probability")
    print(f"   • Perfect for sampling during text generation!")


def main():
    vocab_softmax_example()

    print(f"\n" + "=" * 50)
    print(f"🎓 SUMMARY")
    print(f"=" * 50)
    print(f"Softmax formula: P(word_i) = e^(logit_i) / Σ(e^(logit_j))")
    print(f"")
    print(f"What it does:")
    print(f"1. Converts any real numbers (logits) to probabilities (0-1)")
    print(f"2. All probabilities sum to exactly 1.0")
    print(f"3. Higher logits get exponentially higher probabilities")
    print(f"4. Perfect for next-word prediction in language models!")


if __name__ == "__main__":
    main()
