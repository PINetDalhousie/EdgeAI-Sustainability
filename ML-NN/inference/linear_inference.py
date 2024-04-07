import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

# Load the MNIST test dataset
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0  # Normalize pixel values to [0, 1]
x_test = x_test.reshape(-1, 784).astype(np.float32)  # Reshape to flattened images
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)  # One-hot encode labels

# Load the saved normal model
model = load_model('linear_classifier_model.h5')

# Perform inference on the test data
predictions = model.predict(x_test)

# Convert predictions to class labels (0-9)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate accuracy
true_labels = np.argmax(y_test, axis=1)
accuracy = np.mean(predicted_labels == true_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')
