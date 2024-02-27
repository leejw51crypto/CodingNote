from transformers import AutoTokenizer

# Load a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Example text
text = "Hello, this is a sample text for tokenization."

# Tokenize the text
input_ids = tokenizer.encode(text, add_special_tokens=True)

# Convert token ids to tokens
tokens = tokenizer.convert_ids_to_tokens(input_ids)

print("Token IDs:", input_ids)
print("Tokens:", tokens)
