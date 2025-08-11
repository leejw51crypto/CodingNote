"""
Demo script to visualize the input/output flow in transformer training and inference
Shows the teacher forcing mechanism and autoregressive generation clearly
"""

import torch
import torch.nn as nn


def visualize_teacher_forcing():
    """Demonstrate teacher forcing during training"""
    print("\n" + "=" * 70)
    print("🎓 TEACHER FORCING (Training Mode)")
    print("=" * 70)

    # Example sentence: "I love you"
    print("\n📚 Training Example: English → Italian")
    print("   Source: 'I love you' → 'ti amo'")

    # Tokenized version
    src_tokens = ["<sos>", "I", "love", "you", "<eos>"]
    tgt_tokens = ["<sos>", "ti", "amo", "<eos>"]

    print(f"\n🔤 Tokenized:")
    print(f"   Source: {' '.join(src_tokens)}")
    print(f"   Target: {' '.join(tgt_tokens)}")

    # Teacher forcing steps
    print("\n📖 Teacher Forcing Steps:")
    print("   The model learns to predict the next token given previous tokens")
    print("   " + "-" * 60)

    for i in range(len(tgt_tokens) - 1):
        input_seq = tgt_tokens[: i + 1]
        target_token = tgt_tokens[i + 1]

        print(f"\n   Step {i+1}:")
        print(f"   ➡️  Input:  {' '.join(input_seq):20s} ")
        print(f"   🎯 Target: {target_token:20s}")

        if i == 0:
            print(f"      (Given '<sos>', predict 'ti')")
        elif i == 1:
            print(f"      (Given '<sos> ti', predict 'amo')")
        elif i == 2:
            print(f"      (Given '<sos> ti amo', predict '<eos>')")

    print("\n   " + "-" * 60)
    print("   💡 All predictions happen in parallel during training!")
    print("   The decoder sees: '<sos> ti amo' (shifted right)")
    print("   And predicts:     'ti amo <eos>' (original sequence)")


def visualize_autoregressive_generation():
    """Demonstrate autoregressive generation during inference"""
    print("\n" + "=" * 70)
    print("🤖 AUTOREGRESSIVE GENERATION (Inference Mode)")
    print("=" * 70)

    print("\n📚 Translation Example: English → Italian")
    print("   Source: 'I love you'")
    print("   Goal: Generate 'ti amo'")

    # Source tokens
    src_tokens = ["<sos>", "I", "love", "you", "<eos>"]
    print(f"\n🔤 Source (encoded): {' '.join(src_tokens)}")

    # Generation process
    print("\n📖 Generation Steps (one token at a time):")
    print("   " + "-" * 60)

    generated = ["<sos>"]
    predictions = ["ti", "amo", "<eos>"]

    for i, next_token in enumerate(predictions):
        print(f"\n   Step {i+1}:")
        print(f"   📥 Decoder Input:  {' '.join(generated):25s}")
        print(f"   🔮 Model Predicts: {next_token:25s}")

        generated.append(next_token)

        print(f"   📤 Updated Sequence: {' '.join(generated)}")

        if next_token == "<eos>":
            print(f"   ✅ End of sequence detected - stop generation")
            break
        else:
            print(f"   ↻  Feed back into decoder for next prediction...")

    print("\n   " + "-" * 60)
    print(f"   🎉 Final Translation: {' '.join(generated[1:-1])}")


def visualize_attention_mechanism():
    """Visualize how attention works conceptually"""
    print("\n" + "=" * 70)
    print("🔍 ATTENTION MECHANISM")
    print("=" * 70)

    print("\n📚 Example: Translating 'I love you' → 'ti amo'")

    print("\n1️⃣  Self-Attention in Encoder:")
    print("   Each English word attends to all other English words")
    print("   'love' might strongly attend to 'I' and 'you' for context")

    print("\n2️⃣  Masked Self-Attention in Decoder:")
    print("   Each Italian word can only attend to previous Italian words")
    print("   'amo' can attend to '<sos>' and 'ti', but not future tokens")

    print("\n3️⃣  Cross-Attention (Encoder-Decoder):")
    print("   Each Italian word attends to all English words")
    print("   'ti' might attend strongly to 'you'")
    print("   'amo' might attend strongly to 'love'")

    # Simple attention visualization
    print("\n📊 Simplified Attention Matrix (Cross-Attention):")
    print("        I    love   you")
    print("   ti   0.1   0.2   0.7  → 'ti' focuses on 'you'")
    print("   amo  0.2   0.7   0.1  → 'amo' focuses on 'love'")


def main():
    print("\n" + "🌟" * 35)
    print("  TRANSFORMER MODEL: INPUT/OUTPUT FLOW VISUALIZATION")
    print("🌟" * 35)

    # Show teacher forcing
    visualize_teacher_forcing()

    # Show autoregressive generation
    visualize_autoregressive_generation()

    # Show attention concept
    visualize_attention_mechanism()

    print("\n" + "=" * 70)
    print("📝 KEY DIFFERENCES:")
    print("=" * 70)
    print("\n🎓 Training (Teacher Forcing):")
    print("   • All predictions made in parallel")
    print("   • Model sees correct previous tokens")
    print("   • Fast and efficient")
    print("   • Input: '<sos> ti amo' → Output: 'ti amo <eos>'")

    print("\n🤖 Inference (Autoregressive):")
    print("   • Generate one token at a time")
    print("   • Model sees its own predictions")
    print("   • Slower but necessary for generation")
    print("   • Start with '<sos>', generate until '<eos>'")

    print("\n" + "🌟" * 35)


if __name__ == "__main__":
    main()
