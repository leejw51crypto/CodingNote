import logging

# Suppress warnings from the transformers library
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')

# Prepare input
text = "Hello, this is a simple example to demonstrate how to predict the [MASK] word."
input_ids = tokenizer.encode(text, return_tensors="pt")
print(input_ids)

# Predict masked token with BERT
with torch.no_grad():
    outputs = model(input_ids)
    predictions = outputs.logits

# Get the predicted token ID for the masked token
print(predictions.shape)
print(predictions)
masked_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
print(f"masked_index={masked_index}")
predicted_token_id = predictions[0, masked_index].argmax(axis=1)

#print predicted_token_id
print(predicted_token_id)


# Convert the predicted token ID to a word
predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_id.item())

print("Predicted token:", predicted_token)
