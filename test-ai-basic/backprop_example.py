import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size=3, hidden_size=4, output_size=2):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01  # (3×4)
        self.b1 = np.zeros((1, hidden_size))                       # (1×4)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01 # (4×2)
        self.b2 = np.zeros((1, output_size))                       # (1×2)
        
        # Store dimensions for reference
        self.dims = {
            'input': input_size,    # 3
            'hidden': hidden_size,  # 4
            'output': output_size,  # 2
            'batch': 10            # batch size
        }
        
    def relu(self, Z):
        """ReLU activation: max(0,Z)"""
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        """ReLU derivative: 1 if Z > 0, else 0"""
        return Z > 0
    
    def softmax(self, Z):
        """Softmax activation: exp(Z)/sum(exp(Z))"""
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # Subtract max for numerical stability
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        """
        Forward propagation step with detailed explanations
        Args:
            X: Input data (batch_size × input_size) = (10×3)
        """
        print("\n# Forward Propagation Steps")
        
        # Store input
        self.X = X  # (10×3)
        print("\n## 1. Input Layer")
        print(f"- X shape: `{X.shape}` (batch_size × input_features)")
        print("- Forward equation: `X → Hidden Layer`")
        
        # Step 1: Input to Hidden Layer (Linear)
        print("\n## 2. Hidden Layer Linear Transformation")
        self.Z1 = np.dot(X, self.W1) + self.b1  # (10×3) · (3×4) + (1×4) = (10×4)
        print("- Formula: `Z1 = X·W1 + b1`")
        print(f"- Shapes: `({X.shape}) · ({self.W1.shape}) + ({self.b1.shape}) = ({self.Z1.shape})`")
        print("- Each row of X multiplied by W1 gives features for hidden layer")
        
        # Step 2: Hidden Layer Activation
        print("\n## 3. Hidden Layer Activation")
        self.A1 = self.relu(self.Z1)  # (10×4)
        print("- Formula: `A1 = ReLU(Z1) = max(0, Z1)`")
        print(f"- Shapes: `ReLU({self.Z1.shape}) = {self.A1.shape}`")
        print("- ReLU sets negative values to 0, keeps positive values")
        
        # Step 3: Hidden to Output Layer (Linear)
        print("\n## 4. Output Layer Linear Transformation")
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # (10×4) · (4×2) + (1×2) = (10×2)
        print("- Formula: `Z2 = A1·W2 + b2`")
        print(f"- Shapes: `({self.A1.shape}) · ({self.W2.shape}) + ({self.b2.shape}) = ({self.Z2.shape})`")
        print("- Hidden layer outputs transformed to output layer")
        
        # Step 4: Output Layer Activation
        print("\n## 5. Output Layer Activation (Softmax)")
        self.A2 = self.softmax(self.Z2)  # (10×2)
        print("- Formula: `A2 = softmax(Z2) = exp(Z2)/sum(exp(Z2))`")
        print(f"- Shapes: `softmax({self.Z2.shape}) = {self.A2.shape}`")
        print("- Softmax normalizes outputs to probability distribution")
        
        # Print forward relationships
        print("\n## Forward Relationships")
        print("\n### 1. Layer Connections")
        print("```")
        print("Input → [W1,b1] → Z1 → ReLU → A1 → [W2,b2] → Z2 → Softmax → A2")
        print("```")
        
        print("\n### 2. Activation Functions")
        print("- Hidden Layer: `ReLU(Z1) = max(0, Z1)`")
        print("- Output Layer: `Softmax(Z2) = exp(Z2)/sum(exp(Z2))`")
        
        print("\n### 3. Shape Transformations")
        print("```")
        print(f"Input:        {X.shape}")
        print(f"Hidden Layer: {self.Z1.shape} → {self.A1.shape}")
        print(f"Output Layer: {self.Z2.shape} → {self.A2.shape}")
        print("```")
        
        return self.A2
    
    def backward_propagation(self, Y):
        """
        Backward propagation step with detailed explanations
        Args:
            Y: True labels (batch_size × output_size) = (10×2)
        """
        m = Y.shape[0]  # batch size
        
        print("\n# Backpropagation Steps")
        
        # Step 1: Output Layer Gradient (dZ2)
        self.dZ2 = self.A2 - Y  # (10×2)
        print("\n## 1. Output Layer Gradient (dZ2)")
        print("- Formula: `dZ2 = A2 - Y`  (Combined softmax and cross-entropy derivative)")
        print(f"- Shapes: `({self.A2.shape}) - ({Y.shape}) = ({self.dZ2.shape})`")
        print("- This represents the error at the output layer")
        
        # Step 2: Output Layer Weight Gradients
        self.dW2 = np.dot(self.A1.T, self.dZ2)  # (4×10) · (10×2) = (4×2)
        self.db2 = np.sum(self.dZ2, axis=0, keepdims=True)  # (1×2)
        print("\n## 2. Output Layer Weight Gradients")
        print("- Formula: `dW2 = A1^T · dZ2`")
        print(f"- Shapes: `({self.A1.T.shape}) · ({self.dZ2.shape}) = ({self.dW2.shape})`")
        print("- Formula: `db2 = sum(dZ2, axis=0)`")
        print(f"- Shapes: `sum({self.dZ2.shape}, axis=0) = ({self.db2.shape})`")
        print("- These gradients show how to update W2 and b2")
        
        # Step 3: Hidden Layer Gradient
        self.dA1 = np.dot(self.dZ2, self.W2.T)  # (10×2) · (2×4) = (10×4)
        self.dZ1 = self.dA1 * self.relu_derivative(self.Z1)  # (10×4) ⊙ (10×4) = (10×4)
        print("\n## 3. Hidden Layer Gradient")
        print("- Formula: `dA1 = dZ2 · W2^T`  (Backpropagate error)")
        print(f"- Shapes: `({self.dZ2.shape}) · ({self.W2.T.shape}) = ({self.dA1.shape})`")
        print("- Formula: `dZ1 = dA1 ⊙ relu'(Z1)`  (Apply ReLU derivative)")
        print(f"- Shapes: `({self.dA1.shape}) ⊙ ({self.Z1.shape}) = ({self.dZ1.shape})`")
        print("- Error propagated back through ReLU activation")
        
        # Step 4: Hidden Layer Weight Gradients
        self.dW1 = np.dot(self.X.T, self.dZ1)  # (3×10) · (10×4) = (3×4)
        self.db1 = np.sum(self.dZ1, axis=0, keepdims=True)  # (1×4)
        print("\n## 4. Hidden Layer Weight Gradients")
        print("- Formula: `dW1 = X^T · dZ1`")
        print(f"- Shapes: `({self.X.T.shape}) · ({self.dZ1.shape}) = ({self.dW1.shape})`")
        print("- Formula: `db1 = sum(dZ1, axis=0)`")
        print(f"- Shapes: `sum({self.dZ1.shape}, axis=0) = ({self.db1.shape})`")
        print("- These gradients show how to update W1 and b1")
        
        # Print complete chain rule and relationships
        print("\n## Complete Chain Rule")
        print("```")
        print("For W1: ∂L/∂W1 = ∂L/∂A2 · ∂A2/∂Z2 · ∂Z2/∂A1 · ∂A1/∂Z1 · ∂Z1/∂W1")
        print("For W2: ∂L/∂W2 = ∂L/∂A2 · ∂A2/∂Z2 · ∂Z2/∂W2")
        print("```")
        
        print("\n## Forward-Backward Relationships")
        print("\n### 1. Forward Path")
        print("```")
        print("X → Z1 → A1 → Z2 → A2 → Loss")
        print("```")
        
        print("\n### 2. Backward Path")
        print("```")
        print("Loss → dZ2 → [dW2, dA1] → dZ1 → dW1")
        print("```")
        
        print("\n### 3. Parameter Update vs Backprop Paths")
        print("```")
        print("Parameter Updates: dZ2 → dW2, dZ1 → dW1")
        print("Error Backprop:   dZ2 → dZ1 (using W2, not dW2)")
        print("```")
        
        print("\n### 4. Key Relationships")
        print("```")
        print("Forward: Z1 = X·W1,      Backward: dW1 = X^T·dZ1")
        print("Forward: Z2 = A1·W2,     Backward: dW2 = A1^T·dZ2")
        print("Forward: A1 = ReLU(Z1),  Backward: dZ1 = dA1 ⊙ (Z1 > 0)")
        print("```")
        
        return {
            'dW1': self.dW1, 'db1': self.db1,
            'dW2': self.dW2, 'db2': self.db2
        }

# Example usage
if __name__ == "__main__":
    # Create random data
    X = np.random.randn(10, 3)  # 10 samples, 3 features
    Y = np.random.randint(2, size=(10, 2))  # 10 samples, 2 classes (one-hot encoded)
    
    # Initialize network
    nn = SimpleNeuralNetwork()
    
    # Forward pass with detailed explanation
    print("\n# Forward Propagation")
    output = nn.forward_propagation(X)
    
    # Backward pass with detailed explanation
    print("\n# Backward Propagation")
    gradients = nn.backward_propagation(Y) 