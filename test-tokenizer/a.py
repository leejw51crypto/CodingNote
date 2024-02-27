from transformers import BertTokenizer

# Load the pre-trained tokenizer for the BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Sample text
text = "hello world !"

# Tokenize the text
encoded_input = tokenizer(text, add_special_tokens=True)

# Output the tokenized input
tokens = tokenizer.convert_ids_to_tokens(encoded_input["input_ids"])
print(f"Tokens: {tokens}")
print(f"Token IDs: {encoded_input['input_ids']}")
