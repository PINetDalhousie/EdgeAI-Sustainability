from joblib import dump, load
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import numpy as np

# Load the MNIST dataset
digits = datasets.load_digits()

# Split the data into training and testing sets
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X_train, y_train = data[:n_samples // 2], digits.target[:n_samples // 2]
X_test, y_test = data[n_samples // 2:], digits.target[n_samples // 2:]

# Quantize the data to 16-bit precision
X_train = np.float16(X_train)
X_test = np.float16(X_test)

# Create an SVM classifier
classifier = svm.SVC(gamma=0.001)

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Calculate the accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the trained SVM model
dump(classifier, 'svm_model_quantized.joblib')

# Load the saved SVM classifier
loaded_classifier = load('svm_model_quantized.joblib')

# Make predictions on the test set using the loaded model
y_pred_loaded = loaded_classifier.predict(X_test)

# Calculate the accuracy
accuracy_loaded = metrics.accuracy_score(y_test, y_pred_loaded)
print(f'Accuracy (Loaded Model): {accuracy_loaded * 100:.2f}%')
