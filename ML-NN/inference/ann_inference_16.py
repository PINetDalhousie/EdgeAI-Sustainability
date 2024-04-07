import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
_, (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
x_test = x_test.reshape(x_test.shape[0], -1)

# Load the saved model
loaded_model = load_model('neural_network_model_float16.h5')

# Make predictions on the test data
predictions = loaded_model.predict(x_test)

# Find the predicted labels
predicted_labels = tf.argmax(predictions, axis=1).numpy()

# Calculate the accuracy of the loaded model
accuracy = tf.reduce_mean(tf.cast(predicted_labels == y_test, tf.float32))
print(f'Accuracy (Loaded Model): {accuracy * 100:.2f}%')
