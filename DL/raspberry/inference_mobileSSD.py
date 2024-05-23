import numpy as np
import tensorflow as tf

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
# Perform inference
interpreter.set_tensor(input_details[0]['index'], image_data)
interpreter.invoke()
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

output_data = interpreter.get_tensor(output_details[0]['index'])

predicted_class = np.argmax(output_data)

print("Predicted class:", predicted_class)

phase_name = 'memory_util'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")