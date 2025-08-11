# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational transformer implementation for English-Italian translation with extensive visualization of encoder-decoder interaction, teacher forcing, and autoregressive generation.

## Commands

### Setup
```bash
pip install -r requirements.txt
```

### Training
```bash
python train.py  # Default: 10 epochs, batch size 1
```

### Inference
```bash
python inference.py  # Runs translation with step-by-step visualization
```

### Visualization Tools
```bash
python encoder_decoder_flow.py  # Complete encoder-decoder flow explanation
python demo_transformer_flow.py  # Teacher forcing vs autoregressive comparison
```

## Architecture

### Core Training Flow

The transformer uses an encoder-decoder architecture where:

1. **Encoder** processes source sentence (English) once
2. **Decoder** receives:
   - Encoder output via cross-attention
   - Target sequence SHIFTED RIGHT (teacher forcing)
3. **Output** predicts the ORIGINAL target sequence

Critical insight: During training, decoder input is `[<sos> ti amo]` to predict `[ti amo <eos>]` - the shift enables learning next-token prediction.

### File Structure

**Core Model** (`transformer_model.py`):
- `MultiHeadAttention`: Implements scaled dot-product attention
- `DecoderBlock`: Three sub-layers - masked self-attention, cross-attention, feed-forward
- `Transformer`: Orchestrates encoder → decoder → output projection

**Training** (`train.py`):
- `train_epoch()`: Shows encoder-decoder flow with emojis for clarity
- Teacher forcing with parallel prediction (all tokens at once)
- Displays actual tokens being processed, not just tensor shapes
- Batch size 1 for educational clarity

**Inference** (`inference.py`):
- `translate()`: Step-by-step autoregressive generation
- Shows how decoder builds sequence one token at a time
- Verbose mode displays each generation step

**Data** (`dataset.py`):
- 50 hardcoded English-Italian pairs
- Vocabulary built from training data
- Special tokens: `<sos>`, `<eos>`, `<unk>`, `<pad>`

### Attention Mechanisms

Three types of attention operate in the model:

1. **Encoder Self-Attention**: Q=K=V from encoder (bidirectional)
2. **Decoder Masked Self-Attention**: Q=K=V from decoder (causal mask)
3. **Cross-Attention**: Q from decoder, K=V from encoder

**Cross-Attention Detail** (critical for understanding):
- **Query (Q)**: From decoder - "What am I trying to generate?"
- **Key (K)**: From encoder - "What can I search in the source?"
- **Value (V)**: From encoder - "What information can I retrieve?"

This enables decoder positions to attend to ALL encoder positions, learning translation alignments (e.g., "love" → "amo").

## Model Configuration

Current settings in `train.py`:
- `batch_size`: 1 (line 127)
- `num_epochs`: 10 (line 135)
- `d_model`: 128
- `n_heads`: 4
- `n_layers`: 2
- `learning_rate`: 0.001

## Key Modifications

### Adjust Training
- Line 135 in `train.py`: Increase `num_epochs` for better accuracy
- Line 127 in `train.py`: Change `batch_size` (currently 1 for clarity)

### Control Verbosity
- `transformer_model.py`: Pass `verbose=True` to Transformer constructor for shape printing
- Internal component printing is disabled by default for cleaner output

### Hardware Selection
- Lines 124 (train.py) and 81 (inference.py): Change device from "mps" to "cuda" or "cpu"

## Training vs Inference

**Training (Parallel)**:
- All predictions in one forward pass
- Decoder sees correct previous tokens
- Input: `<sos> ti amo`, Output: `ti amo <eos>`

**Inference (Sequential)**:
- Generate one token at a time
- Decoder sees its own predictions
- Multiple forward passes through decoder

## Visualization Output Examples

Training displays:
```
🌍 Translation Pair:
   English: 'I love you'
   Italian: 'ti amo'

🔄 Encoder-Decoder Flow:
   1️⃣  Encoder processes: <sos> I love you <eos>
   2️⃣  Decoder receives:
        Input (shifted):  <sos> ti amo
        Target (predict): ti amo <eos>
```

Inference displays:
```
📍 Step 1:
   Input to Decoder: <sos>
   Predicted Next:   ti
   → Appending 'ti' and continuing...
```