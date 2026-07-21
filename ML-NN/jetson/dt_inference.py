import joblib
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
import numpy as np
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
_, (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
x_test = x_test.reshape(x_test.shape[0], -1)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# Quantize the data to 16-bit precision
x_test_quantized = np.float16(x_test)

phase_name = 'Load model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the TensorRT engine
with open('decision_tree_model_quantized.trt', 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# Create execution context
context = engine.create_execution_context()

# Allocate device memory
h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float16)
h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.int32)
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)

# Get input and output binding indices
input_idx = engine.get_binding_index("input_1")
output_idx = engine.get_binding_index("output_1")

print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

phase_name = 'Inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Perform inference on the quantized test data
correct_predictions = 0
for i in range(len(x_test)):
    # Copy input data to device
    cuda.memcpy_htod(d_input, np.array(x_test_quantized[i:i + 1], dtype=np.float16))

    # Execute the inference
    context.execute(bindings=[int(d_input), int(d_output)])

    # Copy output data from device
    cuda.memcpy_dtoh(h_output, d_output)

    # Compute accuracy
    predicted_label = h_output[0]
    if predicted_label == y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / len(x_test)
print(f'Accuracy (Loaded Model): {accuracy * 100:.2f}%')
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

phase_name = 'Memory Utilization'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
