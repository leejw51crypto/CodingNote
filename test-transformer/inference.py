import torch

from dataset import get_sample_data
from train import create_padding_mask
from transformer_model import Transformer


def translate(
    model, src_sentence, src_vocab, tgt_vocab, device, max_len=50, verbose=True
):
    model.eval()

    src_tokens = src_sentence.lower().split()
    src_indices = [src_vocab.get(token, src_vocab["<unk>"]) for token in src_tokens]
    src_indices = [src_vocab["<sos>"]] + src_indices + [src_vocab["<eos>"]]
    src_tensor = torch.tensor(src_indices).unsqueeze(0).to(device)

    inv_src_vocab = {v: k for k, v in src_vocab.items()}
    inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}

    if verbose:
        print(f"\n{'='*60}")
        print(f"🌍 Translation Process")
        print(f"{'='*60}")

        src_decoded = [inv_src_vocab.get(idx, "<unk>") for idx in src_indices]
        print(f"\n📥 Source Input: {' '.join(src_decoded)}")
        print(f"   English: '{src_sentence}'")

    src_mask = create_padding_mask(src_tensor)

    with torch.no_grad():
        enc_output = model.encoder(src_tensor, src_mask)
        if verbose:
            print(
                f"\n🔧 Encoder processed source → Hidden states shape: {enc_output.shape}"
            )

    tgt_indices = [tgt_vocab["<sos>"]]

    if verbose:
        print(f"\n🤖 Autoregressive Generation:")
        print(f"{'─'*40}")

    for i in range(max_len):
        tgt_tensor = torch.tensor(tgt_indices).unsqueeze(0).to(device)
        tgt_mask = create_padding_mask(tgt_tensor)

        with torch.no_grad():
            dec_output = model.decoder(tgt_tensor, enc_output, src_mask, tgt_mask)
            output = model.output_layer(dec_output)

        next_token_logits = output[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).item()
        next_token = inv_tgt_vocab.get(next_token_id, "<unk>")

        if verbose:
            current_sequence = [inv_tgt_vocab.get(idx, "<unk>") for idx in tgt_indices]
            print(f"\n📍 Step {i+1}:")
            print(f"   Input to Decoder: {' '.join(current_sequence)}")
            print(f"   Predicted Next:   {next_token}")

            if next_token == "<eos>":
                print(f"   ✅ End of sequence detected")
            else:
                print(f"   → Appending '{next_token}' and continuing...")

        tgt_indices.append(next_token_id)

        if next_token_id == tgt_vocab["<eos>"]:
            break

    translated_tokens = [inv_tgt_vocab.get(idx, "<unk>") for idx in tgt_indices[1:-1]]
    translated_sentence = " ".join(translated_tokens)

    if verbose:
        print(f"\n{'─'*40}")
        print(f"📤 Final Translation: {translated_sentence}")
        print(f"{'='*60}")

    return translated_sentence


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nLoading model...")
    checkpoint = torch.load("best_model.pth", map_location=device)
    src_vocab = checkpoint["src_vocab"]
    tgt_vocab = checkpoint["tgt_vocab"]

    print(f"Model trained for {checkpoint['epoch'] + 1} epochs")
    print(f"Best validation loss: {checkpoint['val_loss']:.4f}")

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_len=100,
        dropout=0.0,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_sentences = [
        "hello",
        "I love you",
        "thank you",
    ]

    print("\n" + "=" * 60)
    print("🌐 TRANSLATION DEMONSTRATIONS")
    print("=" * 60)

    # Show detailed process for first sentence
    print("\n📖 Detailed translation process for first example:")
    translation = translate(
        model, test_sentences[0], src_vocab, tgt_vocab, device, verbose=True
    )

    # Show remaining translations without detailed steps
    print("\n\n📝 Quick translations for remaining examples:")
    print("-" * 60)
    for sentence in test_sentences[1:]:
        translation = translate(
            model, sentence, src_vocab, tgt_vocab, device, verbose=False
        )
        print(f"\n🇬🇧 English: '{sentence}'")
        print(f"🇮🇹 Italian: '{translation}'")

    print("\nNote: The model needs more training epochs for better accuracy.")
    print("Run train.py for longer (increase num_epochs) for better results.")


if __name__ == "__main__":
    main()
