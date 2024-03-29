import numpy as np

# Activation function (ReLU)
def relu(x):
    return np.maximum(0, x)

# Softmax activation function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Input data (shape: 2x3)
X = np.array([[0, 0, 1], [1, 1, 1]])

# Output data (shape: 2x2)
y = np.array([[1, 0], [0, 1]])

print(f"Train input: {X}")
print(f"Train output: {y}")

# Get the true labels from the output data
true_labels = np.argmax(y, axis=1)
print(f"True labels before training: {true_labels}")

# Initialize weights and biases randomly
hidden_weights = np.random.uniform(size=(3, 4))  # shape: 3x4
hidden_bias = np.random.uniform(size=(1, 4))     # shape: 1x4
output_weights = np.random.uniform(size=(4, 2))  # shape: 4x2
output_bias = np.random.uniform(size=(1, 2))     # shape: 1x2

# Training loop
epochs = 5000
learning_rate = 0.1

for i in range(epochs):
    # Forward propagation
    hidden_layer_activation = np.dot(X, hidden_weights) + hidden_bias  # shape: 2x4
    hidden_layer_output = relu(hidden_layer_activation)                # shape: 2x4

    output_layer_activation = np.dot(hidden_layer_output, output_weights) + output_bias  # shape: 2x2
    predicted_output = softmax(output_layer_activation)                                  # shape: 2x2

    # Backpropagation
    error = y - predicted_output  # shape: 2x2
    d_predicted_output = error    # shape: 2x2

    error_hidden_layer = np.dot(d_predicted_output, output_weights.T)  # shape: 2x4
    d_hidden_layer = error_hidden_layer * (hidden_layer_output > 0)   # shape: 2x4

    # Updating weights and biases
    output_weights += learning_rate * np.dot(hidden_layer_output.T, d_predicted_output)  # shape: 4x2
    output_bias += learning_rate * np.sum(d_predicted_output, axis=0, keepdims=True)     # shape: 1x2
    hidden_weights += learning_rate * np.dot(X.T, d_hidden_layer)                        # shape: 3x4
    hidden_bias += learning_rate * np.sum(d_hidden_layer, axis=0, keepdims=True)         # shape: 1x4

    # Print loss every 1000 epochs
    if i % 1000 == 0:
        loss = -np.sum(y * np.log(predicted_output + 1e-7)) / X.shape[0]
        print(f"Epoch: {i}, Loss: {loss}")

# Use the same input data for inference
sample_data = X

# Perform inference on sample data
hidden_layer_activation = np.dot(sample_data, hidden_weights) + hidden_bias  # shape: 2x4
hidden_layer_output = relu(hidden_layer_activation)                          # shape: 2x4

output_layer_activation = np.dot(hidden_layer_output, output_weights) + output_bias  # shape: 2x2
predicted_output = softmax(output_layer_activation)                                  # shape: 2x2

# Get the predicted class labels
predicted_labels = np.argmax(predicted_output, axis=1)

print("Inference on sample data:")
print(sample_data)
print("Predicted labels:")
print(predicted_labels)
print("True labels:")
print(true_labels)
