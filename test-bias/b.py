import numpy as np

# Activation function (ReLU)
def relu(x):
    return np.maximum(0, x)

# Derivative of ReLU
def relu_derivative(x):
    return (x > 0).astype(float)

# Softmax activation function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Cross-entropy loss function
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-7  # Small value to avoid logarithm of zero
    return -np.sum(y_true * np.log(y_pred + epsilon), axis=1)

# Input data (shape: 4x3)
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

# Output data (shape: 4x4)
y = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])

# Initialize weights and biases randomly
hidden_weights = np.random.uniform(size=(3, 5))  # shape: 3x5
hidden_bias = np.random.uniform(size=(1, 5))     # shape: 1x5
output_weights = np.random.uniform(size=(5, 4))  # shape: 5x4
output_bias = np.random.uniform(size=(1, 4))     # shape: 1x4

# Training loop
epochs = 10000
learning_rate = 0.1

for i in range(epochs):
    # Forward propagation
    hidden_layer_activation = np.dot(X, hidden_weights) + hidden_bias  # shape: 4x5
    hidden_layer_output = relu(hidden_layer_activation)                # shape: 4x5

    output_layer_activation = np.dot(hidden_layer_output, output_weights) + output_bias  # shape: 4x4
    predicted_output = softmax(output_layer_activation)                                  # shape: 4x4

    # Backpropagation
    error = y - predicted_output  # shape: 4x4
    d_predicted_output = error    # shape: 4x4

    # Backpropagation to the hidden layer
    error_hidden_layer = np.dot(d_predicted_output, output_weights.T)  # shape: 4x5
    d_hidden_layer = error_hidden_layer * relu_derivative(hidden_layer_output)  # shape: 4x5

    # Updating weights and biases
    output_weights += learning_rate * np.dot(hidden_layer_output.T, d_predicted_output)  # shape: 5x4
    output_bias += learning_rate * np.sum(d_predicted_output, axis=0, keepdims=True)     # shape: 1x4
    hidden_weights += learning_rate * np.dot(X.T, d_hidden_layer)                        # shape: 3x5
    hidden_bias += learning_rate * np.sum(d_hidden_layer, axis=0, keepdims=True)         # shape: 1x5

    # Print loss every 1000 epochs
    if i % 1000 == 0:
        loss = np.mean(cross_entropy_loss(y, predicted_output))
        print(f"Epoch: {i}, Loss: {loss}")

# Generate sample data for inference (shape: 4x3)
sample_data = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])

# Perform inference on sample data
hidden_layer_activation = np.dot(sample_data, hidden_weights) + hidden_bias  # shape: 4x5
hidden_layer_output = relu(hidden_layer_activation)                          # shape: 4x5

output_layer_activation = np.dot(hidden_layer_output, output_weights) + output_bias  # shape: 4x4
predicted_output = softmax(output_layer_activation)                                  # shape: 4x4

print("Inference on sample data:")
print(sample_data)
print("Predicted output:")
print(predicted_output)
