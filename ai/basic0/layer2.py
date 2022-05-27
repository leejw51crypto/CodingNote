import numpy as np

# Define the input data
X = np.array([[0, 1, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1]])

# Define the target data
y = np.array([[0], [1], [1], [0]])

# Define the number of input neurons, hidden neurons, and output neurons
input_neurons = 3
hidden_neurons = 4
output_neurons = 1

# Initialize the weights and biases with random values
weights_hidden = np.random.randn(input_neurons, hidden_neurons)
bias_hidden = np.random.randn(hidden_neurons)
weights_output = np.random.randn(hidden_neurons, output_neurons)
bias_output = np.random.randn(output_neurons)

# Define the learning rate
learning_rate = 0.1

# Train the neural network
for i in range(1000):
    # Forward propagation
    hidden_layer_input = np.dot(X, weights_hidden) + bias_hidden
    hidden_layer_output = 1 / (1 + np.exp(-hidden_layer_input))
    output_layer_input = np.dot(hidden_layer_output, weights_output) + bias_output
    output = 1 / (1 + np.exp(-output_layer_input))

    # Backward propagation
    error_output = output - y
    derivative_output = output * (1 - output)
    derivative_weights_output = np.dot(
        hidden_layer_output.T, error_output * derivative_output
    )
    derivative_bias_output = np.sum(error_output * derivative_output)
    error_hidden = np.dot(error_output * derivative_output, weights_output.T)
    derivative_hidden = hidden_layer_output * (1 - hidden_layer_output)
    derivative_weights_hidden = np.dot(X.T, error_hidden * derivative_hidden)
    derivative_bias_hidden = np.sum(error_hidden * derivative_hidden)

    # Update the weights and biases
    weights_output -= learning_rate * derivative_weights_output
    bias_output -= learning_rate * derivative_bias_output
    weights_hidden -= learning_rate * derivative_weights_hidden
    bias_hidden -= learning_rate * derivative_bias_hidden

# Test the neural network
test_data = np.array([[1, 0, 0], [0, 0, 1]])
hidden_layer_input = np.dot(test_data, weights_hidden) + bias_hidden
hidden_layer_output = 1 / (1 + np.exp(-hidden_layer_input))
output_layer_input = np.dot(hidden_layer_output, weights_output) + bias_output
test_output = 1 / (1 + np.exp(-output_layer_input))
print(test_output)
