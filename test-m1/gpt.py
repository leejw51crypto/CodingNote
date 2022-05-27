from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

def generate_text(prompt):
    device = torch.device("mps")
    print(f"Using device: {device}")

    # Initialize the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to(device)

    # Encode the input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt", padding='max_length', max_length=200).to(device)

    # Create the attention mask
    attention_mask = inputs != tokenizer.pad_token_id

    # Generate text
    outputs = model.generate(inputs, max_length=200, temperature=0.7, num_return_sequences=1, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id)

    # Decode the outputs to get the generated text
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return text

prompt = "In a world with flying cars..."
print(generate_text(prompt))
