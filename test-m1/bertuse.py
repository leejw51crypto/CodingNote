import torch
from transformers import BertForSequenceClassification, AutoTokenizer, BertConfig

device = torch.device('mps')

# Create the BERT model for sequence classification with the correct number of labels
config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = 6  # Set this to the number of labels in your model
model = BertForSequenceClassification(config).to(device) 

# Load the trained model
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


class_labels = {
    0: "Description",
    1: "Entity",
    2: "Abbreviation",
    3: "Human",
    4: "Location",
    5: "Numeric"
}
print(f"class_labels={class_labels}")


def predict(text):
    # Preprocess the text
    tokens = tokenizer(
        text, max_length=512,
        truncation=True, padding='max_length',
        return_tensors='pt'
    )
    
    tokens = {k: v.to(device) for k, v in tokens.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**tokens)
    
    # Get the predicted class
    predicted_class = torch.argmax(outputs.logits).item()
    
    return predicted_class

text = "What does NASA stand for?"
#text = "who is the president of the united states?"
answer_index= predict(text)
print(f"answer_index={answer_index}")
answer_text= class_labels[answer_index];
print(f"question={text}")
print(f"answer_text={answer_text}")