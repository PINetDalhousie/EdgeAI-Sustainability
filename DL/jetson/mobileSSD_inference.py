import numpy as np
import tensorflow as tf
import time
import psutil
import tensorflow.contrib.tensorrt as trt

def get_memory_utilization():
    pid = psutil.Process().pid
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    memory_used = memory_info.rss
    memory_used_mb = memory_used / (1024 * 1024)
    return memory_used_mb

phase_name = 'Load model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='mobilenet_imagenet_subset_final.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

phase_name = 'Load dataset'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load sample image data (replace this with your own image data)
image_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

phase_name = 'inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Perform inference with the TensorFlow Lite model
interpreter.set_tensor(input_details[0]['index'], image_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class_tflite = np.argmax(output_data)
print("Predicted class (TensorFlow Lite):", predicted_class_tflite)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

phase_name = 'Conversion to TensorRT'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Convert TensorFlow Lite model to TensorRT
trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16')

# Write the TensorRT model to a file
with open('mobilenet_imagenet_subset_final_trt.pb', 'wb') as f:
    f.write(trt_graph.SerializeToString())
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

phase_name = 'Load converted TensorRT model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the converted TensorRT model
interpreter_trt = tf.lite.Interpreter(model_path='mobilenet_imagenet_subset_final_trt.pb')
interpreter_trt.allocate_tensors()

# Get input and output tensors for the TensorRT model
input_details_trt = interpreter_trt.get_input_details()
output_details_trt = interpreter_trt.get_output_details()
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

phase_name = 'Inference using converted TensorRT model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Perform inference with the converted TensorRT model
interpreter_trt.set_tensor(input_details_trt[0]['index'], image_data)
interpreter_trt.invoke()
output_data_trt = interpreter_trt.get_tensor(output_details_trt[0]['index'])
predicted_class_trt = np.argmax(output_data_trt)
print("Predicted class (TensorRT):", predicted_class_trt)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# Compare results
print("TensorFlow Lite prediction matches TensorRT prediction:", predicted_class_tflite == predicted_class_trt)

phase_name = 'Memory utilization'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")