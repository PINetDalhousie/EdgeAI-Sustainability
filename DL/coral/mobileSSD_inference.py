import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from edgetpu.detection import BasicEngine
import psutil
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
# Load MNIST test dataset
(_, _), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_test = x_test / 255.0

# Reshape images to (num_samples, height, width, channels)
x_test = np.expand_dims(x_test, axis=-1)

# Convert labels to one-hot encoding
y_test = to_categorical(y_test, 10)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'Load model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the Edge TPU model
model_path = 'mobilenet_mnist_edgetpu.tflite'  # Update to your Edge TPU model path
engine = BasicEngine(model_path)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")


# Function to perform inference on a single image
def predict(image):
    # Resize image for input if needed, MobileSSD typically expects (300, 300, 3)
    image = np.tile(image, (1, 1, 3))  # Repeat grayscale channel for RGB
    input_image = tf.image.resize(image, (300, 300)).numpy()  # Resize to expected dimensions
    input_image = input_image.astype(np.uint8)  # Convert to uint8
    return engine.DetectWithImage(input_image, threshold=0.5, keep_aspect_ratio=True)


# Function to calculate accuracy
def calculate_accuracy(images, labels):
    correct = 0
    total = len(images)
    for i in range(len(images)):
        image = images[i]
        detections = predict(image)
        predicted_label = -1

        if detections:
            predicted_label = detections[0].label  # Assuming single object detection
        true_label = np.argmax(labels[i])

        if predicted_label == true_label:
            correct += 1

    accuracy = correct / total
    return accuracy


# Calculate accuracy on the test dataset
accuracy = calculate_accuracy(x_test, y_test)
print("Accuracy of the Edge TPU model:", accuracy)

phase_name = 'memory_util'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")