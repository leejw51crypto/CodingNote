import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def show_gpt2_logits_and_generation(
    prompt: str, top_k: int = 5, max_new_tokens: int = 10
):
    # ------------------------------
    # 1. Load GPT-2 (small)
    # ------------------------------
    model_name = "gpt2"  # this is the small GPT-2 model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    # GPT-2 has no pad_token by default; avoid warnings
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # ------------------------------
    # 2. Encode prompt
    # ------------------------------
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]  # shape: (1, seq_len)
    attention_mask = inputs["attention_mask"]

    print("=== PROMPT ===")
    print(prompt)
    print()

    # Decode tokens for clarity
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    print("=== INPUT TOKENS ===")
    print("(Ġ = space, Ċ = newline in GPT-2 byte-level BPE)")
    for i, t in enumerate(tokens):
        token_id = input_ids[0, i].item()
        readable = tokenizer.decode([token_id])
        print(f"pos {i:2d}: {t:15s} -> {repr(readable)}")
    print()

    # ------------------------------
    # 3. One forward pass through GPT-2
    # ------------------------------
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # shape: (1, seq_len, vocab_size)

    batch_size, seq_len, vocab_size = logits.shape

    print("=== SHAPES ===")
    print(f"input_ids shape : {input_ids.shape}      # (batch, seq_len)")
    print(f"logits shape    : {logits.shape}   # (batch, seq_len, vocab_size)")
    print("→ seq_len is the SAME for input_ids and logits\n")

    # ------------------------------
    # 4. Inspect per-position predictions
    # ------------------------------
    probs = torch.softmax(logits, dim=-1)  # still (1, seq_len, vocab_size)

    print("=== PER-POSITION NEXT-TOKEN PREDICTIONS (GPT-2) ===")
    for pos in range(seq_len):
        token_here = tokens[pos]
        prob_vec = probs[0, pos]  # (vocab_size,)

        topk = torch.topk(prob_vec, k=top_k)
        top_indices = topk.indices.tolist()
        top_values = topk.values.tolist()
        decoded = tokenizer.convert_ids_to_tokens(top_indices)

        token_readable = tokenizer.decode([input_ids[0, pos].item()])
        print(
            f"\n[Position {pos}] current token: {token_here:15s} -> {repr(token_readable)}"
        )
        print("Top predictions for the *next* token after this position:")
        for rank, (tok_id, tok_prob) in enumerate(
            zip(top_indices, top_values), start=1
        ):
            tok_str = decoded[rank - 1]
            tok_readable = tokenizer.decode([tok_id])
            print(
                f"  {rank}. {tok_str:15s} -> {repr(tok_readable):15s}  prob={tok_prob:.4f}"
            )

    # ------------------------------
    # 5. Simple generation loop using ONLY last logits each step
    # ------------------------------
    print("\n=== SIMPLE GENERATION LOOP (GPT-2, GREEDY) ===")
    generated_ids = input_ids.clone()

    for step in range(max_new_tokens):
        print("=" * 60)
        print(f"STEP {step+1}")
        print("=" * 60)

        # Print input tokens (decoded to readable text)
        print(f"Input tokens ({generated_ids.shape[1]}):")
        for i in range(generated_ids.shape[1]):
            token_id = generated_ids[0, i].item()
            token_text = tokenizer.decode([token_id])
            print(f"  pos {i:2d}: {repr(token_text)}")

        with torch.no_grad():
            out = model(generated_ids)
            step_logits = out.logits  # (1, cur_len, vocab_size)

        # Use ONLY the last position's logits
        last_logits = step_logits[0, -1]  # (vocab_size,)
        probs_last = torch.softmax(last_logits, dim=-1)

        # Greedy decoding: argmax
        next_token_id = (
            torch.argmax(probs_last, dim=-1).unsqueeze(0).unsqueeze(0)
        )  # (1,1)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        next_token_str = tokenizer.decode(next_token_id[0])
        decoded_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Print output logits for each position
        print(f"\nLogits shape: {step_logits.shape}  (batch, seq_len, vocab_size)")
        print(f"Output logits for each position (top 1 prediction):")
        for pos in range(step_logits.shape[1]):
            pos_logits = step_logits[0, pos]
            top_idx = torch.argmax(pos_logits).item()
            top_logit = pos_logits[top_idx].item()
            token_text = tokenizer.decode([top_idx])
            print(f"  pos {pos:2d}: logit={top_logit:8.4f} -> {repr(token_text)}")

        print(f"\nLast position top {top_k}:")
        topk_logits = torch.topk(last_logits, k=top_k)
        for i, (idx, logit_val) in enumerate(
            zip(topk_logits.indices.tolist(), topk_logits.values.tolist())
        ):
            token_text = tokenizer.decode([idx])
            prob_val = probs_last[idx].item()
            print(
                f"  {i+1}. {repr(token_text):15s} logit={logit_val:8.4f}  prob={prob_val:.4f}"
            )

        print(
            f"\nOutput token: {repr(next_token_str)} (id: {next_token_id[0, 0].item()})"
        )
        print(f"Current text: {decoded_text}")
        print()


if __name__ == "__main__":
    prompt = "the dog sat"
    show_gpt2_logits_and_generation(prompt, max_new_tokens=5)
