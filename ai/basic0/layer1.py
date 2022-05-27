import numpy as np

# Define the input data
X = np.array([[0, 1, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1]])

# Define the target data
y = np.array([[0], [1], [1], [0]])

# Define the number of input neurons and output neurons
input_neurons = 3
output_neurons = 1

# Initialize the weights and biases with random values
weights = np.random.randn(input_neurons, output_neurons)
bias = np.random.randn(output_neurons)

# Define the learning rate
learning_rate = 0.1

# Train the neural network
for i in range(1000):
    # Forward propagation
    z = np.dot(X, weights) + bias
    output = 1 / (1 + np.exp(-z))

    # Backward propagation
    error = output - y
    derivative_output = output * (1 - output)
    derivative_weights = np.dot(X.T, error * derivative_output)
    derivative_bias = np.sum(error * derivative_output)

    # Update the weights and biases
    weights -= learning_rate * derivative_weights
    bias -= learning_rate * derivative_bias

# Test the neural network
test_data = np.array([[1, 0, 0], [0, 0, 1]])
test_output = 1 / (1 + np.exp(-(np.dot(test_data, weights) + bias)))
print(test_output)
