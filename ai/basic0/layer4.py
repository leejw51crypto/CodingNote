import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        # initialize weights and biases for the first hidden layer
        self.W1 = np.random.randn(self.input_size, self.hidden_size1)
        self.b1 = np.zeros((1, self.hidden_size1))

        # initialize weights and biases for the second hidden layer
        self.W2 = np.random.randn(self.hidden_size1, self.hidden_size2)
        self.b2 = np.zeros((1, self.hidden_size2))

        # initialize weights and biases for the output layer
        self.W3 = np.random.randn(self.hidden_size2, self.output_size)
        self.b3 = np.zeros((1, self.output_size))

    # define the sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # define the derivative of the sigmoid function
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # define the forward propagation function
    def forward(self, X):
        # compute the activation of the first hidden layer
        hidden_output1 = self.sigmoid(np.dot(X, self.W1) + self.b1)

        # compute the activation of the second hidden layer
        hidden_output2 = self.sigmoid(np.dot(hidden_output1, self.W2) + self.b2)

        # compute the activation of the output layer
        output = self.sigmoid(np.dot(hidden_output2, self.W3) + self.b3)

        return hidden_output1, hidden_output2, output

    # define the backpropagation function
    def backward(self, X, y, hidden_output1, hidden_output2, output):
        # compute the error in the output
        error = y - output

        # compute the gradient of the output layer weights
        output_gradient = error * self.sigmoid_derivative(output)
        dW3 = np.dot(hidden_output2.T, output_gradient)
        db3 = np.sum(output_gradient, axis=0, keepdims=True)

        # compute the gradient of the second hidden layer weights
        hidden_gradient2 = np.dot(output_gradient, self.W3.T) * self.sigmoid_derivative(
            hidden_output2
        )
        dW2 = np.dot(hidden_output1.T, hidden_gradient2)
        db2 = np.sum(hidden_gradient2, axis=0, keepdims=True)

        # compute the gradient of the first hidden layer weights
        hidden_gradient1 = np.dot(
            hidden_gradient2, self.W2.T
        ) * self.sigmoid_derivative(hidden_output1)
        dW1 = np.dot(X.T, hidden_gradient1)
        db1 = np.sum(hidden_gradient1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2, dW3, db3

    # define the training function
    def train(self, X, y, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            # perform forward propagation
            hidden_output1, hidden_output2, output = self.forward(X)

            # perform backpropagation
            dW1, db1, dW2, db2, dW3, db3 = self.backward(
                X, y, hidden_output1, hidden_output2, output
            )

            # update the weights and biases
            self.W1 += learning_rate * dW1
            self.b1 += learning_rate * db1
            self.W2 += learning_rate * dW2
            self.b2 += learning_rate * db2
            self.W3 += learning_rate * dW3
            self.b3 += learning_rate * db3

            # compute the loss
            loss = np.mean(np.square(y - output))

            # print the loss every 100 epochs
            if epoch % 100 == 0:
                print("Epoch:", epoch, "Loss:", loss)


# define the input and output data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# create a neural network with 2 hidden layers
nn = NeuralNetwork(input_size=2, hidden_size1=4, hidden_size2=3, output_size=1)

# train the neural network
nn.train(X, y, num_epochs=10000, learning_rate=0.1)

# test the neural network
hidden_output1, hidden_output2, output = nn.forward(X)
print("Input:\n", X)
print("Output:\n", output.round(2))
