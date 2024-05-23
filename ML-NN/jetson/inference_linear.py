import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import time
import psutil
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

def get_memory_utilization():
    pid = psutil.Process().pid
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    memory_used = memory_info.rss
    memory_used_mb = memory_used / (1024 * 1024)
    return memory_used_mb

phase_name = 'Load dataset'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0  # Normalize pixel values to [0, 1]
x_test = x_test.reshape(-1, 784).astype(np.float32)  # Reshape and cast to float32
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)  # One-hot encode labels
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

phase_name = 'Load model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the TensorRT engine
with open('linear_classifier_model_fp16.trt', 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# Create execution context
context = engine.create_execution_context()

# Allocate device memory
h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)

# Get input and output binding indices
input_idx = engine.get_binding_index("input_1")
output_idx = engine.get_binding_index("Identity")

print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# Initialize accuracy counter
tflite_accuracy = 0.0

phase_name = 'Inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Perform inference on the test data
for i in range(len(x_test)):
    # Copy input data to device
    cuda.memcpy_htod(d_input, np.array(x_test[i:i + 1], dtype=np.float32))

    # Execute the inference
    context.execute(bindings=[int(d_input), int(d_output)])

    # Copy output data from device
    cuda.memcpy_dtoh(h_output, d_output)

    # Compute accuracy
    tflite_accuracy += np.argmax(y_test[i:i + 1], axis=1) == np.argmax(h_output, axis=1)

tflite_accuracy /= len(x_test)
print(f'TFLite Model Accuracy: {tflite_accuracy.item() * 100:.2f}%')
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

phase_name = 'Memory Utilization'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")