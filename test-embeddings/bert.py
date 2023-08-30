from sentence_transformers import SentenceTransformer, util

# Load the pre-trained model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Numerical text
num_text = "block 12345"
# Encode the numerical text to get the embeddings
embedding1 = model.encode(num_text)

print(len(embedding1))

num_text = "block 12346"
# Encode the numerical text to get the embeddings
embedding2 = model.encode(num_text)
print(len(embedding2))


num_text = "block 12300"
# Encode the numerical text to get the embeddings
embedding3 = model.encode(num_text)
print(len(embedding2))

# compute similarity scores of two embeddings
cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)    
print(cosine_scores)


# compute similarity scores of two embeddings
cosine_scores = util.pytorch_cos_sim(embedding1, embedding3)    
print(cosine_scores)