import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

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
model_path = "mobilessd_fp16.tflite"  # Path to your saved TensorFlow Lite MobileNet SSD model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'Load dataset'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load MNIST dataset
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0  # Normalize pixel values to between 0 and 1
x_test = np.expand_dims(x_test, axis=-1).astype(np.float32)  # Add channel dimension
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Preprocess input for TensorFlow Lite model
input_details = interpreter.get_input_details()
input_tensor_index = input_details[0]['index']
input_scale, input_zero_point = input_details[0]['quantization']
x_test_rescaled = x_test / input_scale
x_test_rescaled += input_zero_point
x_test_rescaled = np.clip(x_test_rescaled, 0, 255).astype(np.uint8)

# Perform inference
correct_predictions = 0
total_samples = len(x_test_rescaled)
for i in range(total_samples):
    input_data = np.expand_dims(x_test_rescaled[i], axis=0)
    interpreter.set_tensor(input_tensor_index, input_data)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output_tensor_index = output_details[0]['index']
    output_data = interpreter.get_tensor(output_tensor_index)
    predicted_label = np.argmax(output_data)
    if predicted_label == y_test[i]:
        correct_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_samples

# Print accuracy
print("Accuracy:", accuracy)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'memory_util'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")