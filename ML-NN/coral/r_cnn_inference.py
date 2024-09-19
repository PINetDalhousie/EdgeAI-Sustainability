import numpy as np
import time
import psutil
from PIL import Image
from edgetpu.detection import BasicEngine
from tensorflow.keras.datasets import mnist


def get_memory_utilization():
    pid = psutil.Process().pid
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    memory_used = memory_info.rss
    memory_used_mb = memory_used / (1024 * 1024)
    return memory_used_mb


# Load the MNIST dataset
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0  # Normalize pixel values to [0, 1]
x_test = x_test.reshape(-1, 28, 28, 1).astype(np.float32)  # Reshape and cast to float32
y_test = y_test.astype(np.float32)  # Cast labels to float32

# Load the Edge TPU model
model_path = 'lite_images/r_cnn_model_quantized_edgetpu.tflite'  # Path to your compiled Edge TPU model
engine = BasicEngine(model_path)

# Initialize variables for accuracy calculation
correct_predictions = 0
total_samples = len(x_test)

# Perform inference on the test data
phase_name = 'inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
for i in range(total_samples):
    # Convert the MNIST image to a suitable format for the model
    input_image = np.array(x_test[i] * 255, dtype=np.uint8)  # Scale back to [0, 255]
    input_image = Image.fromarray(input_image.reshape(28, 28)).convert('RGB')  # Convert to RGB

    # Run inference
    detections = engine.DetectWithImage(input_image, threshold=0.5, keep_aspect_ratio=True, relative_coord=True)

    if detections:
        predicted_label = detections[0].label  # Assuming single object detection
    else:
        predicted_label = -1  # No detection

    true_label = int(y_test[i])

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
