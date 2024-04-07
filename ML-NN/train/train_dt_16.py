from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
import joblib
import numpy as np

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten the images
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Quantize the data to 16-bit precision
x_train_quantized = np.float16(x_train)
x_test_quantized = np.float16(x_test)

# Create and train a Decision Tree classifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(x_train_quantized, y_train)

# Save the trained model to a file
joblib.dump(clf, 'decision_tree_model_quantized.joblib')

# Load the saved model for inference
loaded_model = joblib.load('decision_tree_model_quantized.joblib')

# Make predictions on the test data
y_pred = loaded_model.predict(x_test_quantized)

# Calculate the accuracy of the Decision Tree model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy (Loaded Model): {accuracy * 100:.2f}%')
