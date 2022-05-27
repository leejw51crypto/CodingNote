import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # initialize weights and biases for the hidden layer
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))

        # initialize weights and biases for the output layer
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    # define the sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # define the derivative of the sigmoid function
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # define the forward propagation function
    def forward(self, X):
        # compute the activation of the hidden layer
        hidden_output = self.sigmoid(np.dot(X, self.W1) + self.b1)

        # compute the activation of the output layer
        output = self.sigmoid(np.dot(hidden_output, self.W2) + self.b2)

        return hidden_output, output

    # define the backpropagation function
    def backward(self, X, y, hidden_output, output):
        # compute the error in the output
        error = y - output

        # compute the gradient of the output layer weights
        output_gradient = error * self.sigmoid_derivative(output)
        dW2 = np.dot(hidden_output.T, output_gradient)
        db2 = np.sum(output_gradient, axis=0, keepdims=True)

        # compute the gradient of the hidden layer weights
        hidden_gradient = np.dot(output_gradient, self.W2.T) * self.sigmoid_derivative(
            hidden_output
        )
        dW1 = np.dot(X.T, hidden_gradient)
        db1 = np.sum(hidden_gradient, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    # define the training function
    def train(self, X, y, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            # perform forward propagation
            hidden_output, output = self.forward(X)

            # perform backpropagation
            dW1, db1, dW2, db2 = self.backward(X, y, hidden_output, output)

            # update the weights and biases
            self.W1 += learning_rate * dW1
            self.b1 += learning_rate * db1
            self.W2 += learning_rate * dW2
            self.b2 += learning_rate * db2

            # compute the loss
            loss = np.mean(np.square(y - output))

            # print the loss every 100 epochs
            if epoch % 100 == 0:
                print("Epoch:", epoch, "Loss:", loss)


# generate some sample data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# create NeuralNetwork
nn = NeuralNetwork(input_size=2, hidden_size=3, output_size=1)
nn.train(X, y, num_epochs=1000, learning_rate=0.1)
