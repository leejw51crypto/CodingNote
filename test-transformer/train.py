import time

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from transformer_model import Transformer


def create_padding_mask(seq, pad_idx=0):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.uint8)
    return mask == 0


def create_masks(src, tgt, pad_idx=0):
    src_mask = create_padding_mask(src, pad_idx)

    tgt_pad_mask = create_padding_mask(tgt, pad_idx)
    tgt_len = tgt.shape[1]
    tgt_sub_mask = create_look_ahead_mask(tgt_len).to(tgt.device)
    tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(0)

    return src_mask, tgt_mask


def train_epoch(model, dataloader, optimizer, criterion, device, src_vocab, tgt_vocab):
    model.train()
    total_loss = 0

    # Create inverse vocabularies for decoding
    inv_src_vocab = {v: k for k, v in src_vocab.items()}
    inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}

    for batch_idx, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask, tgt_mask = create_masks(src, tgt_input)

        print(f"\n{'='*60}")
        print(f"📦 Batch {batch_idx + 1}")
        print(f"{'='*60}")

        # Display first example in batch for clarity
        if src.shape[0] > 0:
            print(f"\n🔤 Example from batch (first sample):")

            # Decode source
            src_tokens = [inv_src_vocab.get(idx.item(), "<unk>") for idx in src[0]]
            src_text = " ".join(src_tokens)

            # Decode full target
            tgt_full_tokens = [inv_tgt_vocab.get(idx.item(), "<unk>") for idx in tgt[0]]
            tgt_full_text = " ".join(tgt_full_tokens)

            # Decode target input (what decoder sees)
            tgt_input_tokens = [
                inv_tgt_vocab.get(idx.item(), "<unk>") for idx in tgt_input[0]
            ]
            tgt_input_text = " ".join(tgt_input_tokens)

            # Decode target output (what we want to predict)
            tgt_output_tokens = [
                inv_tgt_vocab.get(idx.item(), "<unk>") for idx in tgt[0][1:]
            ]
            tgt_output_text = " ".join(tgt_output_tokens)

            # Extract meaningful parts for display
            src_meaningful = " ".join(
                [t for t in src_tokens if t not in ["<sos>", "<eos>", "<pad>"]]
            )
            tgt_meaningful = " ".join(
                [t for t in tgt_full_tokens if t not in ["<sos>", "<eos>", "<pad>"]]
            )

            print(f"\n🌍 Translation Pair:")
            print(f"   English: '{src_meaningful}'")
            print(f"   Italian: '{tgt_meaningful}'")

            print(f"\n🔄 Encoder-Decoder Flow:")
            print(f"   1️⃣  Encoder processes: {src_text}")
            print(f"        → Creates hidden representations")

            print(f"\n   2️⃣  Decoder receives:")
            print(f"        Input (shifted):  {tgt_input_text}")
            print(f"        Target (predict): {tgt_output_text}")
            print(f"        → Learns to predict each next token in parallel")

        print(f"\n📊 Shapes:")
        print(
            f"   Source: {src.shape} | Target Input: {tgt_input.shape} | Target Output: {tgt_output.shape}"
        )

        optimizer.zero_grad()

        output = model(src, tgt_input, src_mask, tgt_mask)

        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 5 == 0:
            print(f"\n💰 Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask, tgt_mask = create_masks(src, tgt_input)

            output = model(src, tgt_input, src_mask, tgt_mask)

            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 1  # Using batch size 1 for clearer understanding
    d_model = 128
    n_heads = 4
    n_layers = 2
    d_ff = 256
    max_len = 100
    dropout = 0.1
    learning_rate = 0.001
    num_epochs = 10  # Reduced for demo purposes

    print("\nLoading data...")
    train_loader, val_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=batch_size
    )

    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    print("\nInitializing model...")
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    print("\nStarting training...")
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, src_vocab, tgt_vocab
        )
        val_loss = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Time: {epoch_time:.2f}s")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "src_vocab": src_vocab,
                    "tgt_vocab": tgt_vocab,
                },
                "best_model.pth",
            )
            print("Saved best model!")

    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
