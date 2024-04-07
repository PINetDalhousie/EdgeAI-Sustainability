import numpy as np
import tensorflow as tf
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
x_test = x_test / 255.0  # Normalize pixel values to [0, 1]
x_test = x_test.reshape(-1, 28, 28, 1).astype(np.float32)  # Reshape and cast to float32
y_test = y_test.astype(np.float32)  # Cast labels to float32
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")


phase_name = 'Load model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='lite_images/r_cnn_model_fp16.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Initialize variables for accuracy calculation
correct_predictions = 0
total_samples = len(x_test)

phase_name = 'inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Perform inference on the test data
for i in range(len(x_test)):
    interpreter.set_tensor(input_details[0]['index'], x_test[i:i+1])
    interpreter.invoke()
    tflite_predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(tflite_predictions)
    true_label = y_test[i]

    # Check if the prediction matches the true label
    if predicted_label == true_label:
        correct_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_samples * 100
print(f'TFLite Model Accuracy: {accuracy:.2f}%')
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'memory_util'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")