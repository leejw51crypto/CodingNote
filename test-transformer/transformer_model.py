import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Can be used for three different purposes:
    1. Self-attention in encoder: Q=K=V=encoder_states
    2. Masked self-attention in decoder: Q=K=V=decoder_states
    3. Cross-attention in decoder: Q=decoder_states, K=V=encoder_states
    """

    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: Where attention queries come from
                   - Self-attention: same as key/value
                   - Cross-attention: from decoder
            key: What we compute attention scores against
                   - Self-attention: same as query/value
                   - Cross-attention: from encoder
            value: What we actually retrieve/aggregate
                   - Self-attention: same as query/key
                   - Cross-attention: from encoder
            mask: Optional attention mask
        """
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)

        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        output = self.W_o(context)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        x = x.transpose(0, 1)

        for layer in self.layers:
            x = layer(x, mask)

        return x


class DecoderBlock(nn.Module):
    """
    Decoder block with three sub-layers:
    1. Masked self-attention (can't look at future tokens)
    2. Cross-attention (attends to encoder output)
    3. Feed-forward network

    Cross-Attention Flow:
    ┌─────────────┐
    │   Decoder   │ ──── Q (Query) ────┐
    └─────────────┘                    ↓
                                  [Attention]
    ┌─────────────┐                    ↑
    │   Encoder   │ ──── K,V (Key,Value)┘
    └─────────────┘

    This allows decoder to "query" encoder representations
    to understand what to translate.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(DecoderBlock, self).__init__()
        self.masked_attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 1. Masked self-attention: target attends to previous target tokens
        attn_output = self.masked_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. Cross-attention: target attends to all source tokens
        # Query from decoder, Keys/Values from encoder
        attn_output = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # 3. Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        x = x.transpose(0, 1)

        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        return x


class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence tasks.

    Training Flow:
    1. Encoder processes source sequence (e.g., English sentence)
    2. Decoder receives:
       - Encoder output (via cross-attention)
       - Target sequence SHIFTED RIGHT (teacher forcing)
    3. Decoder predicts the ORIGINAL target sequence

    Key insight: During training, decoder input is shifted because
    at position i, we want to predict token i+1 given tokens 0...i

    Example:
        Source: "I love you"
        Target: "ti amo"

        Encoder input:  [<sos>, I, love, you, <eos>]
        Decoder input:  [<sos>, ti, amo]  ← shifted right
        Decoder output: [ti, amo, <eos>]  ← original sequence
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_len=100,
        dropout=0.1,
        verbose=False,
    ):
        super(Transformer, self).__init__()
        self.verbose = verbose
        self.encoder = Encoder(
            src_vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout
        )
        self.decoder = Decoder(
            tgt_vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout
        )
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass through the transformer.

        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len] (shifted during training)
            src_mask: Padding mask for source
            tgt_mask: Causal mask for target (prevents looking ahead)

        Returns:
            Output logits [batch_size, tgt_len, tgt_vocab_size]
        """
        if self.verbose:
            print(f"\n=== Forward pass ===")
            print(f"Source shape: {src.shape}, Target shape: {tgt.shape}")

        # Step 1: Encode source sequence
        enc_output = self.encoder(src, src_mask)

        # Step 2: Decode with cross-attention to encoder output
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)

        # Step 3: Project to vocabulary size
        output = self.output_layer(dec_output)

        if self.verbose:
            print(f"Final output shape: {output.shape}")
            print(f"===================\n")
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
