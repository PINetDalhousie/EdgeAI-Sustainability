import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np

# Load and preprocess the MNIST test dataset
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0  # Normalize pixel values to [0, 1]

# Load the saved model
loaded_model = tf.keras.models.load_model('ffnn_model.h5')

# Perform inference on the test data using the saved model
predictions = loaded_model.predict(x_test)

# Convert predictions to class labels (0-9)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_labels == y_test)
print(f'Saved Model Accuracy: {accuracy * 100:.2f}%')
