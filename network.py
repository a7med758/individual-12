import random  # For shuffling data during mini-batch gradient descent
import numpy as np  # For array and matrix operations

class Network(object):

    def __init__(self, sizes):
        
        self.num_layers = len(sizes)  # Number of layers in the network
        self.sizes = sizes  # List containing the number of neurons in each layer
        # Biases are initialized randomly for each layer except the input layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Weights are initialized randomly between layers; each weight connects two layers
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
       
        for b, w in zip(self.biases, self.weights):  # Iterate through each layer's weights and biases
            a = sigmoid(np.dot(w, a) + b)  # Apply weights, add biases, and apply the sigmoid activation function
        return a  # Return the final output after passing through all layers

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        
        if test_data: n_test = len(test_data)  # If test data is available, get its size
        n = len(training_data)  # Get the number of training samples
        for j in range(epochs):  # Iterate over each epoch
            random.shuffle(training_data)  # Shuffle the training data before creating mini-batches
            # Create mini-batches by slicing the training data
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:  # Update weights and biases for each mini-batch
                self.update_mini_batch(mini_batch, eta)
            if test_data:  # If test data is provided, evaluate and print progress
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, eta):
        
        # Initialize arrays for the gradient of the biases and weights (filled with zeros)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:  # Loop over each training sample in the mini-batch
            # Calculate the gradient for this mini-batch using backpropagation
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # Update the gradient accumulators
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # Update weights and biases by subtracting the scaled gradient
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        
        # Initialize gradients for biases and weights (zero arrays)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Feedforward phase
        activation = x  # Initial input activation (input layer)
        activations = [x]  # List to store activations for each layer
        zs = []  # List to store weighted inputs (z = w*a + b)
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b  # Compute weighted input for the current layer
            zs.append(z)  # Store z-values for later use in backpropagation
            activation = sigmoid(z)  # Apply activation function (sigmoid) to get next layer's activation
            activations.append(activation)  # Store the activation
        # Backpropagation phase
        # Compute the error at the output layer (using cost derivative and sigmoid derivative)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        # Update the gradient for biases and weights at the output layer
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Backpropagate the error to the previous layers
        for l in range(2, self.num_layers):  # Start from the second-to-last layer and move backward
            z = zs[-l]  # Get the weighted input for this layer
            sp = sigmoid_prime(z)  # Compute the derivative of the sigmoid for this layer
            # Compute the error for this layer
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            # Update the gradient for biases and weights for this layer
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)  # Return the gradients for the entire mini-batch

    def evaluate(self, test_data):
        """
        Evaluate the network's performance on the test data.
        Returns the number of test inputs for which the network's output is correct.
        """
        # Use feedforward to get the network's prediction and compare it to the correct label
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        # Count the number of correct predictions
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
    
        #Compute the derivative of the cost function with respect to the output activations.
       
        return (output_activations - y)  # Return the difference between the network output and the true label


# Helper functions for activation and its derivative

def sigmoid(z):
    
    #The sigmoid function, which introduces non-linearity to the network.
    
    return 1.0 / (1.0 + np.exp(-z))  # Compute the sigmoid of z

def sigmoid_prime(z):
    
    #Derivative of the sigmoid function, used during backpropagation.

    return sigmoid(z) * (1 - sigmoid(z))  # Compute the derivative of the sigmoid