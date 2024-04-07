import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0  # Normalize pixel values to [0, 1]
y_test = to_categorical(y_test)

# Reshape the data to add a channel dimension for Conv2D
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Load the saved model for inference
loaded_model = load_model('cnn_model.h5')

# Evaluate the loaded model on the test data
loss, accuracy = loaded_model.evaluate(x_test, y_test, verbose=0)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%')
