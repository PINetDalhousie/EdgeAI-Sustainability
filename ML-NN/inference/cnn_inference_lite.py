import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
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
# Load the MNIST test dataset
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)
y_test = tf.keras.utils.to_categorical(y_test)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'Load model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the quantized TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='lite_images/quantized_cnn_model_fp16.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Initialize accuracy counter
quantized_accuracy = 0

phase_name = 'inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Perform inference on the test dataset
for i in range(len(x_test)):
    input_data = np.array(x_test[i:i + 1], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    quantized_accuracy += np.argmax(y_test[i:i + 1]) == np.argmax(output)

quantized_accuracy /= len(x_test)
print(f'Quantized Model Accuracy (FP16): {quantized_accuracy * 100:.2f}%')
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'memory_util'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")