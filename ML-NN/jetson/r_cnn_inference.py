import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def get_memory_utilization():
    pid = psutil.Process().pid
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    memory_used = memory_info.rss
    memory_used_mb = memory_used / (1024 * 1024)
    return memory_used_mb

def build_trt_engine(model_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.lite.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 30  # 1GB
        builder.fp16_mode = True
        with open(model_path, 'rb') as model:
            parser.parse(model.read())
        return builder.build_cuda_engine(network)

def batch_predict(engine, x_test):
    bindings = [cuda.mem_alloc(x_test.nbytes)]
    cuda.memcpy_htod(bindings[0], x_test)
    with engine.create_execution_context() as context:
        context.execute_v2(bindings=bindings)
        predictions = np.empty([x_test.shape[0], 10], dtype=np.float32)
        cuda.memcpy_dtoh(predictions, bindings[1])
        return predictions

phase_name = 'Load dataset'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0  # Normalize pixel values to [0, 1]
x_test = x_test.reshape(-1, 28, 28, 1).astype(np.float32)  # Reshape and cast to float32
y_test = y_test.astype(np.float32)  # Cast labels to float32
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'Load model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Build TensorRT engine from TensorFlow Lite model
trt_engine = build_trt_engine('lite_images/r_cnn_model_fp16.tflite')
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Initialize variables for accuracy calculation
correct_predictions = 0
total_samples = len(x_test)

phase_name = 'Inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Perform inference on the test data
predictions = batch_predict(trt_engine, x_test)
predicted_labels = np.argmax(predictions, axis=1)
correct_predictions = np.sum(predicted_labels == y_test)
accuracy = correct_predictions / total_samples * 100
print(f'TensorRT Model Accuracy: {accuracy:.2f}%')
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'Memory Utilization'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")