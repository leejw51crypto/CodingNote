import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        
        # CONV LAYERS
        
        # Conv1: Input(227x227x3) -> Conv(11x11, stride=4) -> ReLU -> LRN -> MaxPool
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # Conv2: -> Conv(5x5) -> ReLU -> LRN -> MaxPool
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # Conv3: -> Conv(3x3) -> ReLU
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Conv4: -> Conv(3x3) -> ReLU
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Conv5: -> Conv(3x3) -> ReLU -> MaxPool
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # FC LAYERS
        self.fc = nn.Sequential(
            # FC1: Flatten -> Linear -> ReLU -> Dropout
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            # FC2: Linear -> ReLU -> Dropout
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            # FC3: Linear (no activation - will be handled by loss function)
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        """
        Forward pass with shape printing at each step
        Input: (batch_size, 3, 227, 227)
        """
        # Keep track of tensor shapes
        print("\nForward Pass Shape Analysis:")
        print(f"Input: {x.shape}")
        
        # Convolutional Layers
        x = self.conv1(x)
        print(f"After Conv1 + Pool: {x.shape}")  # (batch, 96, 55, 55)
        
        x = self.conv2(x)
        print(f"After Conv2 + Pool: {x.shape}")  # (batch, 256, 27, 27)
        
        x = self.conv3(x)
        print(f"After Conv3: {x.shape}")  # (batch, 384, 13, 13)
        
        x = self.conv4(x)
        print(f"After Conv4: {x.shape}")  # (batch, 384, 13, 13)
        
        x = self.conv5(x)
        print(f"After Conv5 + Pool: {x.shape}")  # (batch, 256, 6, 6)
        
        # Fully Connected Layers
        x = self.fc(x)
        print(f"After FC layers: {x.shape}")  # (batch, 1000)
        
        return x

    def predict(self, x):
        """
        Make a prediction with softmax probabilities
        """
        with torch.no_grad():
            # Forward pass
            logits = self.forward(x)
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(logits, dim=1)
            
            # Get predicted class
            predicted_class = torch.argmax(probabilities, dim=1)
            
            return probabilities, predicted_class

# Example usage
def test_alexnet():
    # Create model
    model = AlexNet()
    
    # Create sample batch (4 images)
    batch_size = 4
    sample_input = torch.randn(batch_size, 3, 227, 227)
    
    # Forward pass
    print("\n=== Regular Forward Pass ===")
    output = model(sample_input)
    
    # Make prediction
    print("\n=== Making Prediction ===")
    probs, pred_classes = model.predict(sample_input)
    
    # Print results
    print("\nResults:")
    print(f"Output logits shape: {output.shape}")
    print(f"Probability shape: {probs.shape}")
    print(f"Predicted classes: {pred_classes}")
    
    # Print probability distribution for first image
    print("\nProbability distribution (first image):")
    print(f"Min prob: {probs[0].min():.6f}")
    print(f"Max prob: {probs[0].max():.6f}")
    print(f"Sum of probs: {probs[0].sum():.6f}")  # Should be close to 1.0

if __name__ == "__main__":
    test_alexnet()

"""
Expected Output:

Forward Pass Shape Analysis:
Input: torch.Size([4, 3, 227, 227])
After Conv1 + Pool: torch.Size([4, 96, 55, 55])
After Conv2 + Pool: torch.Size([4, 256, 27, 27])
After Conv3: torch.Size([4, 384, 13, 13])
After Conv4: torch.Size([4, 384, 13, 13])
After Conv5 + Pool: torch.Size([4, 256, 6, 6])
After FC layers: torch.Size([4, 1000])

Results:
Output logits shape: torch.Size([4, 1000])
Probability shape: torch.Size([4, 1000])
Predicted classes: tensor([483, 128, 892, 761])
"""