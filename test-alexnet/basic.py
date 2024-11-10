import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

"""
Dataset Specification:
    Training Data:
        - 200 points total (100 per class)
        - Class 0: Cluster centered at (-2, -2) with 0.5 std deviation
        - Class 1: Cluster centered at (2, 2) with 0.5 std deviation
    
Input Specification:
    - Input shape: (batch_size, 2)
    - Each input is a 2D point: [x, y]
    - Feature ranges: Typically between -4 and 4 for both x and y
    - Example inputs: [-2,-2], [2,2], [0,0], etc.

Output Specification:
    - Output shape: (batch_size, 1)
    - Binary classification: 0 or 1
    - Raw output: Probability between 0 and 1
    - Threshold at 0.5 for final class prediction

Classification:
    - Class 0: Points near (-2, -2) - "negative" class
    - Class 1: Points near (2, 2) - "positive" class
"""

###################
# 1. Create Data
###################
def create_dataset(n_samples=100):
    # Create two clusters: class 0 centered at (-2,-2), class 1 at (2,2)
    class0 = torch.normal(-2, 0.5, (n_samples//2, 2))
    class1 = torch.normal(2, 0.5, (n_samples//2, 2))
    
    X = torch.vstack([class0, class1])
    y = torch.hstack([torch.zeros(n_samples//2), torch.ones(n_samples//2)])
    
    return X, y.reshape(-1, 1)

###################
# 2. Define Model
###################
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Smaller architecture with regularization
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.2),       # Add dropout to prevent overfitting
            
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

###################
# 3. Training
###################
def train_model(model, X, y, epochs=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)  # Add L2 regularization
    loss_fn = nn.BCELoss()
    
    for epoch in range(epochs):
        model.train()  # Set to training mode (enables dropout)
        
        # Forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if epoch % 100 == 0:
            model.eval()  # Set to evaluation mode for reporting
            with torch.no_grad():
                val_loss = loss_fn(model(X), y)
                print(f'Epoch {epoch}, Training Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')

###################
# 4. Inference
###################
def predict(model, x):
    model.eval()
    with torch.no_grad():
        prob = model(x)
        pred = prob > 0.5
        return pred, prob

###################
# Run Everything
###################
if __name__ == "__main__":
    # Create data
    X, y = create_dataset(200)
    
    # Show dataset examples
    print("\nDataset Examples:")
    print("Point           | Class")
    print("-" * 25)
    for i in range(5):
        print(f"[{X[i][0]:5.2f}, {X[i][1]:5.2f}] | {int(y[i][0])}")
    print("...")
    for i in range(-5, 0):
        print(f"[{X[i][0]:5.2f}, {X[i][1]:5.2f}] | {int(y[i][0])}")
    
    print(f"\nDataset Statistics:")
    print(f"Total points: {len(X)}")
    print(f"Class 0 points: {torch.sum(y == 0).item()}")
    print(f"Class 1 points: {torch.sum(y == 1).item()}")
    print(f"Feature ranges:")
    print(f"  X1: [{X[:,0].min():.2f}, {X[:,0].max():.2f}]")
    print(f"  X2: [{X[:,1].min():.2f}, {X[:,1].max():.2f}]")
    
    # Create and train model
    print("\nTraining Model:")
    model = BinaryClassifier()
    train_model(model, X, y)
    
    # Test various points
    print("\nExample Classifications:")
    print("Input Point      | Predicted Class | Probability")
    print("-" * 45)
    
    test_points = torch.tensor([
        [-2.0, -2.0],  # Should be Class 0
        [2.0, 2.0],    # Should be Class 1
        [0.0, 0.0],    # Decision boundary
        [-1.0, 2.0],   # Mixed coordinates
        [3.0, -3.0],   # Far from training data
    ])
    
    for point in test_points:
        pred, prob = predict(model, point.reshape(1, -1))
        print(f"[{point[0]:5.1f}, {point[1]:5.1f}] | Class {int(pred[0][0])}         | {prob[0][0]:.3f}")

    # Add visualization code
    def plot_decision_boundary():
        # Create a mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # Get predictions for all grid points
        grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
        Z, probs = predict(model, grid_points)
        Z = Z.reshape(xx.shape)
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        
        # Plot training points
        class0_mask = y.flatten() == 0
        class1_mask = y.flatten() == 1
        
        plt.scatter(X[class0_mask, 0], X[class0_mask, 1], 
                   c='blue', label='Class 0', edgecolors='k')
        plt.scatter(X[class1_mask, 0], X[class1_mask, 1], 
                   c='red', label='Class 1', edgecolors='k')
        
        # Plot test points with their predictions
        test_predictions = []
        for point in test_points:
            pred, prob = predict(model, point.reshape(1, -1))
            pred_class = int(pred[0][0])
            prob_val = float(prob[0][0])
            test_predictions.append((pred_class, prob_val))
            
            # Add text annotation showing prediction and probability
            plt.scatter(point[0], point[1], c='green', marker='*', 
                       s=200, edgecolors='k')
            plt.annotate(f'Class {pred_class}\n({prob_val:.2f})', 
                        (point[0], point[1]), 
                        xytext=(10, 10), 
                        textcoords='offset points',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Binary Classification Results\nGreen stars: Test points with predicted class')
        plt.legend(['Class 0', 'Class 1'])
        plt.grid(True)
        plt.show()

    # Call the plotting function after training
    plot_decision_boundary()
