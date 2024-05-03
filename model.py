import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, activation='sigmoid', weight_scale=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.weight_scale = weight_scale

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * self.weight_scale
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * self.weight_scale
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def bipolar_sigmoid(self, x):
        return 2 / (1 + np.exp(-x)) - 1

    def bipolar_sigmoid_derivative(self, x):
        return 0.5 * (1 + x) * (1 - x)

    def activation_function(self, x):
        if self.activation == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation == 'bipolar_sigmoid':
            return self.bipolar_sigmoid(x)
        else:
            raise ValueError("Invalid activation function")

    def activation_derivative(self, x):
        if self.activation == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.activation == 'bipolar_sigmoid':
            return self.bipolar_sigmoid_derivative(x)
        else:
            raise ValueError("Invalid activation function")

    def forward(self, X):
        # Input to hidden layer
        self.hidden_output = self.activation_function(np.dot(X, self.weights_input_hidden) + self.bias_hidden)

        # Hidden to output layer
        self.output = self.activation_function(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output)

        return self.output

    def backward(self, X, y, output):
        # Calculate error
        error = y - output

        # Backpropagation
        d_output = error * self.activation_derivative(output)
        error_hidden = d_output.dot(self.weights_hidden_output.T)
        d_hidden = error_hidden * self.activation_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * self.learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += X.T.dot(d_hidden) * self.learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        return self.forward(X)
