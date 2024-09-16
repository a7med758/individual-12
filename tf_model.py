import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# Load and preprocess the MNIST data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Flatten the images and normalize pixel values to [0, 1]
train_images = train_images.reshape((60000, 28*28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28*28)).astype('float32') / 255

# Convert labels to one-hot encoding
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Build the neural network model
model = Sequential([
    Dense(30, activation='sigmoid', input_shape=(28*28,)),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=SGD(), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=30, batch_size=10, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')
