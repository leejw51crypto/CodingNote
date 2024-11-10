import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define constants
INPUT_SIZE = 28 * 28  # MNIST images are 28x28
HIDDEN_SIZE = 128
NUM_CLASSES = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 5

# Custom Dataset class
class MNISTDataset(Dataset):
    def __init__(self, num_samples=1000):
        # Generate dummy data instead of loading real MNIST
        self.images = torch.randn(num_samples, INPUT_SIZE)  # Random images
        self.labels = torch.randint(0, NUM_CLASSES, (num_samples,))  # Random labels
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Neural Network Model
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
        )
    
    def forward(self, x):
        return self.network(x)

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 100 == 99:
            print(f'Batch: {batch_idx + 1}, Loss: {running_loss/100:.3f}, '
                  f'Accuracy: {100.*correct/total:.2f}%')
            running_loss = 0.0

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return val_loss / len(val_loader), accuracy

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create datasets and dataloaders
    train_dataset = MNISTDataset(num_samples=5000)
    val_dataset = MNISTDataset(num_samples=1000)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model, loss function, and optimizer
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_val_accuracy = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch + 1}/{NUM_EPOCHS}')
        print('-' * 60)
        
        # Train for one epoch
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        print(f'Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_accuracy:.2f}%')
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saved new best model with accuracy: {best_val_accuracy:.2f}%')

if __name__ == '__main__':
    main()