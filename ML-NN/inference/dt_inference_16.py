import joblib
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
import numpy as np
import time


def get_memory_utilization():
    pid = psutil.Process().pid
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    memory_used = memory_info.rss
    memory_used_mb = memory_used / (1024 * 1024)
    return memory_used_mb


phase_name = 'Load dataset'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the MNIST dataset
_, (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
x_test = x_test.reshape(x_test.shape[0], -1)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")


# Quantize the data to 16-bit precision
x_test_quantized = np.float16(x_test)

phase_name = 'Load model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the saved Decision Tree model
loaded_model = joblib.load('decision_tree_model_quantized.joblib')
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Make predictions on the quantized test data
y_pred = loaded_model.predict(x_test_quantized)

# Calculate the accuracy of the loaded Decision Tree model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy (Loaded Model): {accuracy * 100:.2f}%')
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'memory_util'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")