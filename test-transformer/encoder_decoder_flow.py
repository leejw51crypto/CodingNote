"""
Comprehensive visualization of encoder-decoder interaction in transformers
Shows how source and target sentences flow through the model during training and inference
"""

import torch


def visualize_training_flow():
    """Show the complete training flow with encoder and decoder"""
    print("\n" + "=" * 80)
    print("🎓 COMPLETE TRAINING FLOW (Encoder-Decoder Architecture)")
    print("=" * 80)

    # Example translation
    print("\n📚 Example: English → Italian")
    print("   'I love you' → 'ti amo'")

    # Step 1: Encoder
    print("\n" + "─" * 70)
    print("1️⃣  ENCODER PROCESSING (Source Language)")
    print("─" * 70)

    src_tokens = ["<sos>", "I", "love", "you", "<eos>"]
    print(f"\n📥 Encoder Input: {' '.join(src_tokens)}")
    print("\n🔄 Processing:")
    print("   Each token → Embedding → Add Position → Self-Attention → Feed-Forward")
    print("\n📤 Encoder Output: Hidden states for each position")
    print("   Position 0: Hidden(<sos>)  [128-dim vector]")
    print("   Position 1: Hidden(I)      [128-dim vector]")
    print("   Position 2: Hidden(love)   [128-dim vector]")
    print("   Position 3: Hidden(you)    [128-dim vector]")
    print("   Position 4: Hidden(<eos>)  [128-dim vector]")

    # Step 2: Decoder
    print("\n" + "─" * 70)
    print("2️⃣  DECODER PROCESSING (Target Language)")
    print("─" * 70)

    tgt_full = ["<sos>", "ti", "amo", "<eos>"]
    tgt_input = ["<sos>", "ti", "amo"]  # Shifted right
    tgt_output = ["ti", "amo", "<eos>"]  # Original

    print(f"\n📥 Original Target: {' '.join(tgt_full)}")
    print(f"📥 Decoder Input:   {' '.join(tgt_input)}  ← Shifted right!")
    print(f"🎯 Expected Output: {' '.join(tgt_output)} ← What we predict")

    print("\n🔄 Processing at Each Position (IN PARALLEL):")
    print("   " + "─" * 60)

    for i in range(len(tgt_input)):
        visible_tokens = tgt_input[: i + 1]
        predict_token = tgt_output[i]

        print(f"\n   Position {i+1}:")
        print(f"   👁️  Can see: {' '.join(visible_tokens)}")
        print(f"   🔗 Attends to ALL encoder outputs via cross-attention")
        print(f"   🎯 Predicts: '{predict_token}'")

        if i == 0:
            print(f"      • Self-attention: Only sees '<sos>'")
            print(f"      • Cross-attention: Sees all of 'I love you'")
            print(f"      • Likely 'you' → 'ti' mapping via attention")
        elif i == 1:
            print(f"      • Self-attention: Sees '<sos> ti'")
            print(f"      • Cross-attention: Sees all of 'I love you'")
            print(f"      • Likely 'love' → 'amo' mapping via attention")
        else:
            print(f"      • Self-attention: Sees '<sos> ti amo'")
            print(f"      • Cross-attention: Sees all of 'I love you'")
            print(f"      • Detects sequence completion → '<eos>'")

    print("\n   " + "─" * 60)
    print("   ⚡ All three predictions happen in ONE forward pass!")
    print("   🎭 Masked attention prevents looking at future tokens")


def visualize_attention_connections():
    """Visualize how attention connects encoder and decoder"""
    print("\n" + "=" * 80)
    print("🔗 ATTENTION CONNECTIONS")
    print("=" * 80)

    print("\n📊 Three Types of Attention in Transformer:")

    print("\n1️⃣  ENCODER SELF-ATTENTION")
    print("   " + "─" * 40)
    print("   'I' ←→ 'love' ←→ 'you'")
    print("   Each word sees ALL other words (bidirectional)")
    print("   Builds contextual understanding of source")

    print("\n2️⃣  DECODER MASKED SELF-ATTENTION")
    print("   " + "─" * 40)
    print("   '<sos>' → 'ti' → 'amo' → '<eos>'")
    print("   Each word sees ONLY previous words (causal/masked)")
    print("   Maintains autoregressive property")

    print("\n3️⃣  ENCODER-DECODER CROSS-ATTENTION")
    print("   " + "─" * 40)
    print("   Q (Query) source:     Decoder (what am I generating?)")
    print("   K (Key) source:       Encoder (what can I search?)")
    print("   V (Value) source:     Encoder (what can I retrieve?)")
    print("")
    print("   Example: Generating 'ti'")
    print("   • Query from decoder: 'What Italian word for position 1?'")
    print("   • Keys from encoder:  ['I', 'love', 'you']")
    print("   • Values from encoder: [Hidden(I), Hidden(love), Hidden(you)]")
    print("   • Result: Strong attention to 'you' → generates 'ti'")

    # Attention matrix visualization
    print("\n📈 Cross-Attention Weight Matrix (Q×K^T):")
    print("         ENCODER (K,V)")
    print("            I     love    you")
    print("   D  ti   0.15   0.25   0.60  ← Q from decoder")
    print("   E  amo  0.20   0.70   0.10  ← Q from decoder")
    print("   C <eos> 0.33   0.33   0.34  ← Q from decoder")
    print("   O")
    print("   D")
    print("   E")
    print("   R")
    print("")
    print("   Each decoder position (Q) queries all encoder positions (K)")
    print("   to retrieve weighted combination of encoder values (V)")


def visualize_inference_flow():
    """Show how inference differs from training"""
    print("\n" + "=" * 80)
    print("🤖 INFERENCE FLOW (Step-by-Step Generation)")
    print("=" * 80)

    print("\n📚 Same Example: 'I love you' → ?")

    print("\n1️⃣  ENCODER (Runs Once)")
    print("   " + "─" * 40)
    print("   Input: '<sos> I love you <eos>'")
    print("   Output: Hidden states [cached for reuse]")

    print("\n2️⃣  DECODER (Runs Multiple Times)")
    print("   " + "─" * 40)

    steps = [
        ("Step 1", ["<sos>"], "ti"),
        ("Step 2", ["<sos>", "ti"], "amo"),
        ("Step 3", ["<sos>", "ti", "amo"], "<eos>"),
    ]

    for step_name, decoder_input, predicted in steps:
        print(f"\n   {step_name}:")
        print(f"   📥 Decoder sees: {' '.join(decoder_input)}")
        print(f"   🔗 Cross-attends to encoder outputs")
        print(f"   🎯 Generates: '{predicted}'")
        if predicted != "<eos>":
            print(f"   ↻  Append and continue...")
        else:
            print(f"   ✅ Stop generation")


def compare_training_vs_inference():
    """Direct comparison of training vs inference"""
    print("\n" + "=" * 80)
    print("⚖️  TRAINING vs INFERENCE COMPARISON")
    print("=" * 80)

    print("\n📊 Key Differences:")
    print("\n┌─────────────────┬────────────────────────┬────────────────────────┐")
    print("│                 │ TRAINING               │ INFERENCE              │")
    print("├─────────────────┼────────────────────────┼────────────────────────┤")
    print("│ Encoder Runs    │ Once per batch         │ Once per sequence      │")
    print("│ Decoder Runs    │ Once (parallel)        │ N times (sequential)   │")
    print("│ Decoder Input   │ Full target (shifted)  │ Generated tokens       │")
    print("│ Speed           │ Fast (parallel)        │ Slower (sequential)    │")
    print("│ Teacher Forcing │ Yes (sees correct)     │ No (sees predictions)  │")
    print("│ Attention Mask  │ Causal mask            │ Causal mask            │")
    print("│ Batch Processing│ Multiple sequences     │ Usually one sequence   │")
    print("└─────────────────┴────────────────────────┴────────────────────────┘")

    print("\n🎯 Training (One Forward Pass):")
    print("   Source: [<sos> I love you <eos>]  →  Encoder  →  Hidden States")
    print("                                           ↓")
    print("   Target: [<sos> ti amo]          →  Decoder  →  [ti amo <eos>]")
    print("           (input - shifted)                      (output - original)")

    print("\n🎯 Inference (Multiple Forward Passes):")
    print("   Source: [<sos> I love you <eos>]  →  Encoder  →  Hidden States")
    print("                                           ↓")
    print("   Step 1: [<sos>]         →  Decoder  →  ti")
    print("   Step 2: [<sos> ti]      →  Decoder  →  amo")
    print("   Step 3: [<sos> ti amo]  →  Decoder  →  <eos>")


def main():
    print("\n" + "🌟" * 40)
    print("  TRANSFORMER ENCODER-DECODER: COMPLETE FLOW VISUALIZATION")
    print("🌟" * 40)

    # Show complete training flow
    visualize_training_flow()

    # Show attention connections
    visualize_attention_connections()

    # Show inference flow
    visualize_inference_flow()

    # Compare training vs inference
    compare_training_vs_inference()

    print("\n" + "=" * 80)
    print("💡 KEY INSIGHTS:")
    print("=" * 80)
    print("\n1. Training is NOT 'input: source, output: target'")
    print("   It's: 'encoder: source, decoder: shifted target → original target'")
    print("\n2. The shift is crucial for learning to predict next tokens")
    print("\n3. Cross-attention allows decoder to 'look at' source while generating")
    print("\n4. Training uses parallel prediction (fast)")
    print("   Inference uses sequential generation (necessary)")
    print("\n5. Both encoder AND decoder are active during training")
    print("   The encoder provides context, decoder learns to translate")

    print("\n" + "🌟" * 40)


if __name__ == "__main__":
    main()
