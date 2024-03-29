import numpy as np

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Loss function and its derivative
def cross_entropy(predicted, true):
    return -np.sum(true * np.log(predicted))

def delta_cross_entropy(predicted, true):
    return predicted - true

# Initialize weights and biases
np.random.seed(0)  # For reproducibility
weights1 = np.random.rand(4, 8)  # First layer weights
biases1 = np.random.rand(1, 8)   # First layer biases
weights2 = np.random.rand(8, 2)  # Second layer weights
biases2 = np.random.rand(1, 2)   # Second layer biases

print(f"weights1 shape: {weights1.shape}")
print(f"biases1 shape: {biases1.shape}")
print(f"weights2 shape: {weights2.shape}")
print(f"biases2 shape: {biases2.shape}")

# Example input and true output (batch of 2 examples)
X = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # Input data
Y = np.array([[0, 1], [1, 0]])              # True labels

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")

# Forward Pass
hidden_input = np.dot(X, weights1) + biases1  # To hidden layer
hidden_output = sigmoid(hidden_input)         # Activation
final_input = np.dot(hidden_output, weights2) + biases2  # To output layer
final_output = softmax(final_input)           # Softmax

print(f"hidden_input shape: {hidden_input.shape}")
print(f"hidden_output shape: {hidden_output.shape}")
print(f"final_input shape: {final_input.shape}")
print(f"final_output shape: {final_output.shape}")

# Compute loss
loss = cross_entropy(final_output, Y)
print(f"Initial Loss: {loss}")

# Back-propagation
# Output layer gradients
d_loss_output = delta_cross_entropy(final_output, Y)
d_weights2 = np.dot(hidden_output.T, d_loss_output)
d_biases2 = np.sum(d_loss_output, axis=0, keepdims=True)

print(f"d_loss_output shape: {d_loss_output.shape}")
print(f"d_weights2 shape: {d_weights2.shape}")
print(f"d_biases2 shape: {d_biases2.shape}")

# Hidden layer gradients
d_hidden_output = np.dot(d_loss_output, weights2.T)
d_hidden_input = d_hidden_output * sigmoid_derivative(hidden_output)
d_weights1 = np.dot(X.T, d_hidden_input)
d_biases1 = np.sum(d_hidden_input, axis=0, keepdims=True)

print(f"d_hidden_output shape: {d_hidden_output.shape}")
print(f"d_hidden_input shape: {d_hidden_input.shape}")
print(f"d_weights1 shape: {d_weights1.shape}")
print(f"d_biases1 shape: {d_biases1.shape}")

# Update weights and biases
learning_rate = 0.1
weights2 -= learning_rate * d_weights2
biases2 -= learning_rate * d_biases2
weights1 -= learning_rate * d_weights1
biases1 -= learning_rate * d_biases1

print(f"Updated biases1: {biases1}")
print(f"Updated biases2: {biases2}")
