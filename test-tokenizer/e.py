from transformers import Word2VecTokenizer, Word2VecModel

# Example sentence
sentence = "This is a sample sentence demonstrating tokenization and word embeddings."

# Load pre-trained Word2Vec tokenizer
tokenizer = Word2VecTokenizer.from_pretrained("word2vec-google-news-300")

# Tokenize the sentence
tokens = tokenizer.tokenize(sentence)
print("Tokens:", tokens)

# Convert tokens to token IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Token IDs:", token_ids)

# Load pre-trained Word2Vec model
model = Word2VecModel.from_pretrained("word2vec-google-news-300")

# Get embeddings for the token IDs
outputs = model(input_ids=token_ids)
word_embeddings = outputs.last_hidden_state
print("Word embeddings shape:", word_embeddings.shape)
