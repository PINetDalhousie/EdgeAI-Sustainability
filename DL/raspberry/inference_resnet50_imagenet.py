import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
def get_memory_utilization():
    # Get the process ID of the current script
    pid = psutil.Process().pid

    # Get process object for the current script
    process = psutil.Process(pid)

    # Get memory utilization
    memory_info = process.memory_info()

    # Memory utilization in bytes
    memory_used = memory_info.rss

    # Convert bytes to megabytes for better readability
    memory_used_mb = memory_used / (1024 * 1024)

    return memory_used_mb


phase_name = 'Load dataset'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load MNIST dataset
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
# Load the TensorFlow Lite model
tflite_model_path = 'resnet50_imagenet.tflite'
# tflite_model_path = 'resnet50_mnist.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Function to perform inference on a single image
def predict(image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Function to calculate accuracy
def calculate_accuracy(images, labels):
    correct = 0
    total = len(images)
    for i in range(len(images)):
        image = np.expand_dims(images[i], axis=0).astype(np.float32)
        predicted = np.argmax(predict(image))
        true_label = np.argmax(labels[i])
        if predicted == true_label:
            correct += 1
    accuracy = correct / total
    return accuracy

phase_name = 'inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Calculate accuracy on the test dataset
accuracy = calculate_accuracy(x_test, y_test)
print("Accuracy of the TensorFlow Lite model:", accuracy)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'memory_util'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")