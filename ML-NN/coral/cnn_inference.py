import numpy as np
import time
import psutil
import tflite_runtime.interpreter as tflite
from tensorflow.keras.datasets import mnist


def get_memory_utilization():
    pid = psutil.Process().pid
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    memory_used_mb = memory_info.rss / (1024 * 1024)
    return memory_used_mb


phase_name = 'Load dataset'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the MNIST dataset
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)
y_test = tf.keras.utils.to_categorical(y_test)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

phase_name = 'Load Edge TPU model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the quantized TensorFlow Lite model compiled for the Edge TPU
interpreter = tflite.Interpreter(model_path='quantized_cnn_model_edgetpu.tflite',
                                 experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

phase_name = 'Inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
correct_predictions = 0

# Perform inference on the test dataset
for i in range(len(x_test)):
    input_data = np.array(x_test[i:i + 1], dtype=np.uint8)  # Use uint8 for quantized models
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output and calculate accuracy
    output = interpreter.get_tensor(output_details[0]['index'])
    correct_predictions += np.argmax(y_test[i]) == np.argmax(output)

# Calculate accuracy
accuracy = correct_predictions / len(x_test)
print(f'Edge TPU Quantized Model Accuracy: {accuracy * 100:.2f}%')
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

phase_name = 'Memory utilization'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
