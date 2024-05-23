import tensorflow as tf
import numpy as np

def get_memory_utilization():
    # Get the process ID of the current script
    pid = psutil.Process().pid

    # Get process object for the current script
    process = psutil.Process(pid)

    # Get memory utilization
    memory_info = process.memory_info()

    # Memory utilization in bytes
    memory_used = memory_info.rss

    # Convert bytes to megabytes for better readability
    memory_used_mb = memory_used / (1024 * 1024)

    return memory_used_mb

phase_name = 'Load model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the TensorFlow Lite model
tflite_model_path = 'resnet50_mnist.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'Load data'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Load sample image data (replace this with your own image data)
sample_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Perform inference
interpreter.set_tensor(input_details[0]['index'], sample_image)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Post-process the output if needed
predicted_class = np.argmax(output_data)

print("Predicted class:", predicted_class)

print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'memory_util'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")