import torch
import numpy as np
from transformers import BertTokenizer, BertModel

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embedding(sentence):
    """
    Generate embeddings for a given sentence using the BERT model.
    
    Args:
    - sentence (str): The input sentence for which to get the embeddings.
    
    Returns:
    - numpy.array: The embeddings for each token in the sentence (excluding [CLS] and [SEP] tokens).
    """
    
    # Tokenize and encode the sentence
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state

    # Get embeddings for each token, ignoring [CLS] and [SEP]
    return embeddings[0, 1:-1].numpy()

# Given sequence
sequence = ["Hello", "how", "are", "you", "?"]
sequence_embedding = get_embedding(' '.join(sequence))

# Printing the tokens and the shape of their embeddings
print(sequence)
print(f"Sequece embedding shape: {sequence_embedding.shape}")

# Given batch of sequences
batch = [
    ["Hello", "how", "are", "you", "?"],
    ["I'm", "fine", "thank", "you", "."],
    ["What", "is", "your", "name", "?"]
]

print(f"Batch :{batch}")
# iterate batch , print
for seq in batch:
    print(f"{'_'.join(seq)} : {get_embedding(' '.join(seq)).shape}")
    
# Get embeddings for each sequence in the batch
batch_embeddings = [get_embedding(' '.join(seq)) for seq in batch]


# Find max length among all embeddings to use for padding
max_length = max(embedding.shape[0] for embedding in batch_embeddings)
print(f"Maximum sequence length in the batch: {max_length}")

# Pad all embeddings to the max length to ensure they have uniform shape
padded_embeddings = [np.pad(embedding, ((0, max_length - embedding.shape[0]), (0, 0)), mode='constant') for embedding in batch_embeddings]

# Convert list of padded embeddings to a numpy array for easier processing
batch_embeddings_array = np.stack(padded_embeddings)
print(f"Batch embeddings shape: {batch_embeddings_array.shape}")
