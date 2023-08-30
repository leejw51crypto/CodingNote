import openai
import torch
from scipy.spatial.distance import cosine
import os

# Load the pre-trained model
# read env OPENAI_API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")
model = "text-similarity-davinci-001"
#model = "text-embedding-ada-002"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_embedding(text):
    response = openai.Embedding.create(model=model, input=[text])
    embedding = response['data'][0]['embedding']
    return torch.tensor(embedding).to(device)

# Numerical text
num_text1 = "block 12345"
num_text2 = "block 12346"
num_text3 = "block 12343"

# Get the embeddings
embedding1 = get_embedding(num_text1)
embedding2 = get_embedding(num_text2)
embedding3 = get_embedding(num_text3)

# Compute cosine similarity
cosine_similarity = 1 - cosine(embedding1.cpu().numpy(), embedding2.cpu().numpy())
print(cosine_similarity)

cosine_similarity = 1 - cosine(embedding1.cpu().numpy(), embedding3.cpu().numpy())
print(cosine_similarity)
