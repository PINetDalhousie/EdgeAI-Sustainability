import numpy as np
import time
import psutil
from PIL import Image
from edgetpu.detection import BasicEngine
from sklearn import datasets


def get_memory_utilization():
    pid = psutil.Process().pid
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    memory_used = memory_info.rss
    memory_used_mb = memory_used / (1024 * 1024)
    return memory_used_mb


# Load the MNIST dataset
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X_test, y_test = data[n_samples // 2:], digits.target[n_samples // 2:]

# Load the Edge TPU model
model_path = 'lite_images/svm_model_quantized_edgetpu.tflite'  # Path to your compiled Edge TPU model
engine = BasicEngine(model_path)

# Initialize variables for accuracy calculation
correct_predictions = 0
total_samples = len(X_test)

# Perform inference on the test data
phase_name = 'inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
for i in range(total_samples):
    # Convert the MNIST image to a suitable format for the model
    input_image = (X_test[i] * 255).astype(np.uint8)  # Scale to [0, 255]
    input_image = Image.fromarray(input_image.reshape(8, 8)).convert('RGB')  # Convert to RGB

    # Run inference
    detections = engine.DetectWithImage(input_image, threshold=0.5, keep_aspect_ratio=True, relative_coord=True)

    if detections:
        predicted_label = detections[0].label  # Assuming single object detection
    else:
        predicted_label = -1  # No detection

    true_label = y_test[i]

    # Check if the prediction matches the true label
    if predicted_label == true_label:
        correct_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_samples * 100
print(f'Edge TPU Model Accuracy: {accuracy:.2f}%')
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Memory utilization
phase_name = 'memory_util'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
