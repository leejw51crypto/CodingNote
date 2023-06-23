import numpy as np

class AffineLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros(output_size)
        self.inputs = None
        self.weights_gradient = None
        self.bias_gradient = None

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.bias

    def backward(self, grad_output, learning_rate):
        self.weights_gradient = np.dot(self.inputs.T, grad_output)
        self.bias_gradient = np.sum(grad_output, axis=0)
        grad_input = np.dot(grad_output, self.weights.T)
        self.weights -= learning_rate * self.weights_gradient
        self.bias -= learning_rate * self.bias_gradient
        return grad_input


# Example usage
# Create an instance of the AffineLayer with input size of 2 and output size of 3
layer = AffineLayer(2, 3)

# Forward pass
x = np.array([[1, 2]])
output = layer.forward(x)
print("Forward pass output:")
print(output)

# Backward pass
grad_output = np.ones_like(output)
learning_rate = 0.1
grad_input = layer.backward(grad_output, learning_rate)
print("Backward pass gradient:")
print(grad_input)

# Updated weights and bias
print("Updated weights:")
print(layer.weights)
print("Updated bias:")
print(layer.bias)
