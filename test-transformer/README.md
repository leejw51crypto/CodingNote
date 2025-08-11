# Transformer Model - Educational Implementation

A simplified transformer implementation for English-Italian translation, designed to clearly visualize the encoder-decoder flow and understand how transformers work.

## Key Features

- 🎯 **Clear visualization** of teacher forcing vs autoregressive generation
- 📊 **Batch size 1** for easier understanding of data flow
- 🔍 **Detailed logging** showing encoder-decoder interaction
- 🚀 **MPS support** for Apple Silicon acceleration

## Understanding the Flow

### Training (Teacher Forcing)

During training, the model processes data in **one forward pass**:

```
Source: "I love you"
Target: "ti amo"

ENCODER receives: [<sos> I love you <eos>]
         ↓ (creates hidden representations)
         
DECODER receives:
  - Input:  [<sos> ti amo]     ← shifted right
  - Output: [ti amo <eos>]     ← predicts original
  
All predictions happen in parallel!
```

### Inference (Autoregressive)

During inference, generation happens **step by step**:

```
Step 1: [<sos>]        → generates → ti
Step 2: [<sos> ti]     → generates → amo  
Step 3: [<sos> ti amo] → generates → <eos>
```

## Files Overview

### Core Implementation
- `transformer_model.py` - Complete transformer with encoder, decoder, and attention
- `dataset.py` - 50 English-Italian phrase pairs for quick experimentation
- `train.py` - Training loop with clear visualization of teacher forcing
- `inference.py` - Step-by-step autoregressive generation

### Visualization Scripts
- `encoder_decoder_flow.py` - Comprehensive flow visualization
- `demo_transformer_flow.py` - Teacher forcing vs autoregressive comparison

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train the Model
```bash
python train.py
```

### Run Inference
```bash
python inference.py
```

### Understand the Flow
```bash
# See complete encoder-decoder interaction
python encoder_decoder_flow.py

# Compare training vs inference
python demo_transformer_flow.py
```

## Key Concepts Illustrated

### 1. The Shift in Decoder Input
- **Why?** At position i, we predict token i+1
- **Training:** Decoder sees correct previous tokens (teacher forcing)
- **Inference:** Decoder sees its own predictions (autoregressive)

### 2. Three Types of Attention
1. **Encoder Self-Attention:** Bidirectional, builds context
2. **Decoder Masked Self-Attention:** Causal, prevents looking ahead
3. **Cross-Attention:** Connects decoder to encoder output

### 3. Parallel vs Sequential Processing
- **Training:** All positions predicted simultaneously (fast)
- **Inference:** One token at a time (necessary for generation)

## Model Configuration

- Embedding dimension: 128
- Attention heads: 4
- Layers: 2 (encoder & decoder)
- Feed-forward dimension: 256
- Batch size: 1 (for clarity)
- Vocabulary: ~90 English, ~95 Italian tokens

## Output Examples

### Training Output
```
============================================================
📦 Batch 1
============================================================

🌍 Translation Pair:
   English: 'I love you'
   Italian: 'ti amo'

🔄 Encoder-Decoder Flow:
   1️⃣  Encoder processes: <sos> I love you <eos>
        → Creates hidden representations

   2️⃣  Decoder receives:
        Input (shifted):  <sos> ti amo
        Target (predict): ti amo <eos>
        → Learns to predict each next token in parallel
```

### Inference Output
```
🤖 Autoregressive Generation:
────────────────────────────────────────

📍 Step 1:
   Input to Decoder: <sos>
   Predicted Next:   ti
   → Appending 'ti' and continuing...

📍 Step 2:
   Input to Decoder: <sos> ti
   Predicted Next:   amo
   → Appending 'amo' and continuing...
```

## Tips for Learning

1. **Start with `encoder_decoder_flow.py`** to understand the architecture
2. **Run training with small epochs** to see the flow without waiting
3. **Modify batch size** in train.py to see multiple examples
4. **Add print statements** to trace tensor shapes through the model

## Note

This is an educational implementation optimized for understanding, not production use. The model needs significant training (50+ epochs) for reasonable translation quality.