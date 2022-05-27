# Import necessary modules
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from time import time
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import BertForSequenceClassification, BertConfig

# Set the device to MPS for efficient computation on macOS
device = torch.device('mps')
print("device=",device)

# Load the first 1000 examples from the TREC dataset for training
#trec = load_dataset("trec", split='train[:1000]')
trec = load_dataset("trec", split='train')
print(trec)

# Initialize the tokenizer from 'bert-base-uncased' model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the texts to a maximum length of 512 with truncation and padding
tokens = tokenizer(
    trec['text'], max_length=512,
    truncation=True, padding='max_length'
)

print(f"trect={trec}")
print(f"coarse_label={trec['coarse_label']}")
print(f"fine_label={trec['fine_label']}")

# Dictionary mapping class indices to labels
class_labels = {
    0: "Description",
    1: "Entity",
    2: "Abbreviation",
    3: "Human",
    4: "Location",
    5: "Numeric"
}
print(f"class_labels={class_labels}")

# One-hot encode the labels in the dataset
labels = np.zeros(
    (len(trec), max(trec['coarse_label'])+1)
)
labels[np.arange(len(trec)), trec['coarse_label']] = 1

# Define the custom Dataset class
class TrecDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    # This method is called when an item is requested
    def __getitem__(self, idx):
        input_ids = self.tokens[idx].ids
        attention_mask = self.tokens[idx].attention_mask
        labels = self.labels[idx]
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels, dtype=torch.float32)
        }

    # This method returns the total number of items
    def __len__(self):
        return len(self.labels)

# Instantiate the Dataset and DataLoader
dataset = TrecDataset(tokens, labels)
loader = torch.utils.data.DataLoader(dataset, batch_size=64)

# Create the BERT model for sequence classification with the correct number of labels
config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = max(trec['coarse_label'])+1
model = BertForSequenceClassification(config).to(device) 

# Freeze the weights of the pre-trained BERT model
for param in model.bert.parameters():
    param.requires_grad = False

# Set the model to training mode
model.train()

# Initialize the Adam optimizer with a learning rate of 5e-5
optim = torch.optim.Adam(model.parameters(), lr=5e-5)

# Initialize the list to hold the times for each loop
loop_time = []

# Set up the training loop with a progress bar
loop = tqdm(loader, leave=True)
for batch in loop:
    # Move batch to the device
    batch_mps = {
        'input_ids': batch['input_ids'].to(device),
        'attention_mask': batch['attention_mask'].to(device),
        'labels': batch['labels'].to(device)
    }

    t0 = time()

    # Clear the gradients from the last step
    optim.zero_grad()

    # Forward pass through the model
    outputs = model(**batch_mps)

    # Extract the loss
    loss = outputs[0]

    # Backpropagate the loss to compute gradients
    loss.backward()

    # Update the model parameters
    optim.step()

    # Record the time for the loop
    loop_time.append(time()-t0)

    # Print the loss to the progress bar
    loop.set_postfix(loss=loss.item())  

torch.save(model.state_dict(), 'model.pth')


# First, we need to load the test data
trec_test = load_dataset("trec", split='test')

# Tokenize the test data
tokens_test = tokenizer(
    trec_test['text'], max_length=512,
    truncation=True, padding='max_length', return_tensors='pt'
)

# Create the test dataset and dataloader
dataset_test = TrecDataset(tokens_test, trec_test['coarse_label'])
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=64)

# Set the model to evaluation mode
model.eval()

# Initialize counters
total, correct = 0, 0



def one_hot_encode(labels, num_classes):
    one_hot = torch.zeros(labels.size(0), num_classes).to(labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1).long(), 1)  # Convert labels to torch.int64
    return one_hot



with torch.no_grad():
    for batch in tqdm(loader_test):
        batch = {k: v.to(device) for k, v in batch.items()}
        batch['labels'] = one_hot_encode(batch['labels'], num_classes=config.num_labels)  # convert labels to one-hot
        outputs = model(**batch)
        predicted = outputs.logits

        # Update counters
        total += batch['labels'].size(0)
        correct += (torch.argmax(predicted, dim=1) == torch.argmax(batch['labels'], dim=1)).sum().item()



# Compute accuracy
accuracy = correct / total

print(f'Test accuracy: {accuracy * 100:.2f}%')