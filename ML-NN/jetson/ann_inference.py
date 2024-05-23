import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import numpy as np
import time
import psutil
import onnx
import tf2onnx
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

def build_engine(onnx_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 30  # 1GB
        builder.max_batch_size = 1

        with open(onnx_file_path, 'rb') as model:
            parser.parse(model.read())

        return builder.build_cuda_engine(network)

def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream

def do_inference(context, bindings, inputs, outputs, stream):
    cuda.memcpy_htod_async(bindings[0], inputs, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs, bindings[1], stream)
    stream.synchronize()
    return outputs

# Logging and Timing
phase_name = 'Load dataset'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
x_test = x_test.reshape(x_test.shape[0], -1)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

phase_name = 'Load model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
loaded_model = load_model('neural_network_model_float16.h5')
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

phase_name = 'Convert to ONNX'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
onnx_model_path = "model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(loaded_model, output_path=onnx_model_path)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

phase_name = 'Convert to TensorRT'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
engine = build_engine(onnx_model_path)
context = engine.create_execution_context()
h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

phase_name = 'Inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
correct_predictions = 0
for i in range(len(x_test)):
    input_data = np.asarray(x_test[i], dtype=np.float32)
    np.copyto(h_input, input_data)
    bindings = [int(d_input), int(d_output)]
    outputs = do_inference(context, bindings, h_input, h_output, stream)
    predicted_label = np.argmax(outputs)
    if predicted_label == y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / len(x_test)
print(f'Accuracy (TensorRT Model): {accuracy * 100:.2f}%')
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

phase_name = 'Memory Utilization'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")