import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten the images
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Quantize the data to 16-bit precision
x_train_quantized = np.float16(x_train)
x_test_quantized = np.float16(x_test)

# Initialize and train the KNN classifier
k = 3  # You can choose the number of neighbors (k) based on your needs
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train_quantized, y_train)

# Make predictions on the test data
y_pred = knn.predict(x_test_quantized)

# Calculate the accuracy of the KNN model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the trained KNN model to a file
joblib.dump(knn, 'knn_model_quantized.pkl')

# Load the saved model for inference
loaded_knn = joblib.load('knn_model_quantized.pkl')

# Example: Make predictions using the loaded model
new_data = np.random.random((1, 784))  # Replace with your input data
new_data_quantized = np.float16(new_data)
predicted_label = loaded_knn.predict(new_data_quantized)
print(f'Predicted label: {predicted_label[0]}')
