import numpy as np
import tensorflow as tf
import time
import psutil

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
tflite_model_path = 'resnet50_imagenet.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'Load data'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

image_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'Inference with TensorFlow Lite model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Perform inference with the TensorFlow Lite model
interpreter.set_tensor(input_details[0]['index'], image_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output_data)
print("Predicted class (TensorFlow Lite model):", predicted_class)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Transformation to TensorRT model
phase_name = 'Transformation to TensorRT model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

converted_model = tf.experimental.tensorrt.Converter(...).convert()
converted_model.save('resnet50_tensorrt')
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Inference with the TensorRT model
phase_name = 'Inference with TensorRT model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

tensorrt_model = tf.saved_model.load('resnet50_tensorrt')
output_data_tensorrt = tensorrt_model(image_data)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Logging memory utilization
phase_name = 'Memory utilization'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
