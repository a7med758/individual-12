import network
import mnist_loader

# Load the MNIST data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Create the network with a specified architecture
net = network.Network([784, 30, 10])

# Train the network
net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)

# Evaluate the network's performance
accuracy = net.evaluate(test_data)
print(f"Test accuracy: {accuracy}")