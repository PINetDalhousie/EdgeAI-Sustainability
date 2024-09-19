import numpy as np
import tflite_runtime.interpreter as tflite
from tensorflow.keras.datasets import mnist
import psutil
import time


def get_memory_utilization():
    pid = psutil.Process().pid
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    memory_used = memory_info.rss
    memory_used_mb = memory_used / (1024 * 1024)
    return memory_used_mb


# Load dataset
phase_name = 'Load dataset'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0  # Normalize pixel values to [0, 1]
x_test = x_test.reshape(-1, 784).astype(np.float32)  # Reshape to 784 and cast to float32
y_test = np.eye(10)[y_test]  # One-hot encode labels
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Load TFLite model and allocate tensors
phase_name = 'Load model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the Edge TPU-compiled model
interpreter = tflite.Interpreter(
    model_path="linear_classifier_model_edgetpu.tflite",
    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Initialize accuracy counter
tflite_accuracy = 0.0

# Perform inference on the test data
phase_name = 'inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
for i in range(len(x_test)):
    # Prepare input tensor and invoke the interpreter
    input_data = np.expand_dims(x_test[i], axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output tensor
    tflite_predictions = interpreter.get_tensor(output_details[0]['index'])

    # Compare predictions with the ground truth
    if np.argmax(tflite_predictions, axis=1) == np.argmax(y_test[i]):
        tflite_accuracy += 1

# Calculate final accuracy
tflite_accuracy /= len(x_test)
print(f'TFLite Model Accuracy on Coral: {tflite_accuracy * 100:.2f}%')
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Memory utilization
phase_name = 'memory_util'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
