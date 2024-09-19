import numpy as np
import time
import psutil
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from tensorflow.keras.datasets import mnist


def get_memory_utilization():
    pid = psutil.Process().pid
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    memory_used = memory_info.rss
    memory_used_mb = memory_used / (1024 * 1024)
    return memory_used_mb


phase_name = 'Load dataset'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the MNIST test dataset
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0  # Normalize pixel values to [0, 1]
x_test = x_test.reshape(-1, 28, 28, 1).astype(np.float32)  # Reshape for the model
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)  # One-hot encode labels
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'Load model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the Edge TPU model using pycoral
interpreter = make_interpreter('model_edgetpu.tflite')  # Compiled model for TPU
interpreter.allocate_tensors()

# Get input/output details from the interpreter
input_details = common.input_details(interpreter)
output_details = common.output_details(interpreter)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Initialize accuracy counter
tpu_accuracy = 0.0

phase_name = 'inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Perform inference on the test data
for i in range(len(x_test)):
    # Preprocess and set input tensor
    common.set_input(interpreter, x_test[i:i + 1])

    # Run inference
    interpreter.invoke()

    # Get the output tensor (predictions)
    tpu_predictions = common.output_tensor(interpreter, output_details['index'])

    # Calculate accuracy
    tpu_accuracy += np.argmax(y_test[i:i + 1], axis=1) == np.argmax(tpu_predictions, axis=1)

# Calculate and print overall accuracy
tpu_accuracy /= len(x_test)
accuracy_percentage = float(tpu_accuracy) * 100
print(f'Edge TPU Model Accuracy: {accuracy_percentage:.2f}%')
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'memory_util'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
