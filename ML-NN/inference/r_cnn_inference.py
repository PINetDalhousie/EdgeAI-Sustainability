import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Load the MNIST test dataset
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0  # Normalize pixel values to [0, 1]
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')  # Reshape and cast to float32
y_test = to_categorical(y_test, num_classes=10)  # One-hot encode labels

# Load the saved normal (non-quantized) model
model = load_model('r_cnn_model.h5')

# Evaluate the model on the test data
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Accuracy: {accuracy * 100:.2f}%')
