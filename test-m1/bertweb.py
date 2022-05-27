from transformers import BertForSequenceClassification, BertTokenizer
import torch


# Define the model name
model_name = 'bert-base-uncased'

# Load the pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Your class labels
class_labels = {
    0: "Description",
    1: "Entity",
    2: "Abbreviation",
    3: "Human",
    4: "Location",
    5: "Numeric"
}

def predict(text):
    # Preprocess the text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    print(f"inputs={inputs}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted class
    predicted_class = torch.argmax(outputs.logits).item()
    
    return predicted_class


text = "what NASA stand for?"
answer_index= predict(text)
print(f"answer_index={answer_index}")
answer_text= class_labels[answer_index];
print(f"question={text}")
print(f"answer_text={answer_text}")