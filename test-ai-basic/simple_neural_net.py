import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def softmax(self, x):
        # Numerical stability: subtract max
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        # Store intermediate values for backpropagation
        self.X = X
        
        # First layer: Z1 = X·W1 + b1
        self.Z1 = np.dot(X, self.W1) + self.b1
        print(f"\nForward Pass:")
        print(f"Z1 shape: {self.Z1.shape} = X({X.shape}) · W1({self.W1.shape}) + b1({self.b1.shape})")
        
        # Apply ReLU: A1 = max(0, Z1)
        self.A1 = np.maximum(0, self.Z1)
        print(f"A1 shape: {self.A1.shape} (ReLU activation)")
        
        # Second layer: Z2 = A1·W2 + b2
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        print(f"Z2 shape: {self.Z2.shape} = A1({self.A1.shape}) · W2({self.W2.shape}) + b2({self.b2.shape})")
        
        # Apply softmax: A2 = softmax(Z2)
        self.A2 = self.softmax(self.Z2)
        print(f"A2 shape: {self.A2.shape} (Softmax activation)")
        
        return self.A2

    def backward(self, X, Y, learning_rate=0.01):
        m = X.shape[0]  # batch size
        
        print(f"\nBackward Pass:")
        # 1. Output Layer Gradient (∂L/∂Z2)
        dZ2 = self.A2 - Y
        print(f"dZ2 shape: {dZ2.shape} = A2({self.A2.shape}) - Y({Y.shape})")
        
        # 2. Output Weights Gradient (∂L/∂W2)
        dW2 = np.dot(self.A1.T, dZ2)
        print(f"dW2 shape: {dW2.shape} = A1.T({self.A1.T.shape}) · dZ2({dZ2.shape})")
        
        # 3. Output Bias Gradient (∂L/∂b2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        print(f"db2 shape: {db2.shape} = sum(dZ2)")
        
        # 4. Hidden Layer Gradient (∂L/∂A1)
        dA1 = np.dot(dZ2, self.W2.T)
        print(f"dA1 shape: {dA1.shape} = dZ2({dZ2.shape}) · W2.T({self.W2.T.shape})")
        
        # 5. ReLU Gradient (∂L/∂Z1)
        dZ1 = dA1 * (self.Z1 > 0)
        print(f"dZ1 shape: {dZ1.shape} = dA1 * (Z1 > 0)")
        
        # 6. Input Weights Gradient (∂L/∂W1)
        dW1 = np.dot(X.T, dZ1)
        print(f"dW1 shape: {dW1.shape} = X.T({X.T.shape}) · dZ1({dZ1.shape})")
        
        # 7. Input Bias Gradient (∂L/∂b1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        print(f"db1 shape: {db1.shape} = sum(dZ1)")

        # Update parameters
        print("\nParameter Updates:")
        print(f"W2({self.W2.shape}) -= {learning_rate} * dW2({dW2.shape})")
        print(f"W1({self.W1.shape}) -= {learning_rate} * dW1({dW1.shape})")
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def cross_entropy_loss(self, Y_true, Y_pred):
        m = Y_true.shape[0]
        # Add small epsilon to avoid log(0)
        log_probs = -np.log(Y_pred + 1e-10)
        loss = np.sum(Y_true * log_probs) / m
        return loss

# Example usage
def main():
    # Create dummy data
    X = np.random.randn(10, 3)  # 10 samples, 3 features
    Y = np.eye(2)[np.random.randint(0, 2, 10)]  # One-hot encoded labels for 2 classes

    # Initialize model
    model = SimpleNeuralNetwork(input_size=3, hidden_size=4, output_size=2)

    # Training loop
    for epoch in range(100):
        # Forward pass
        output = model.forward(X)
        
        # Calculate loss
        loss = model.cross_entropy_loss(Y, output)
        
        # Backward pass
        model.backward(X, Y)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

if __name__ == "__main__":
    main() 