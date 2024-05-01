from transformers import BertTokenizer

# Load a pre-trained tokenizer (e.g., BERT tokenizer)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example sentence
sentence = "This is a sample sentence."

# Tokenize the sentence
tokens = tokenizer.tokenize(sentence)
print("Tokens:", tokens)

# Convert tokens to indices
indices = tokenizer.convert_tokens_to_ids(tokens)
print("Indices:", indices)
