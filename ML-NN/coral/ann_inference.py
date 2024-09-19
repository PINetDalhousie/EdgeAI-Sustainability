import numpy as np
import time
import psutil
import tflite_runtime.interpreter as tflite
from tensorflow.keras.datasets import mnist


def get_memory_utilization():
    # Get the process ID of the current script
    pid = psutil.Process().pid
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    memory_used_mb = memory_info.rss / (1024 * 1024)
    return memory_used_mb


phase_name = 'Load dataset'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the MNIST dataset
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32)  # Reshape for the model input
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

phase_name = 'Load Edge TPU model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the compiled Edge TPU model
interpreter = tflite.Interpreter(model_path='quantized_neural_network_model_fp16_edgetpu.tflite',
                                 experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

phase_name = 'Inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
correct_predictions = 0

# Run inference on test data using the Edge TPU
for i in range(len(x_test)):
    input_shape = input_details[0]['shape']
    input_data = x_test[i].reshape(input_shape).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get predictions
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(predictions)

    # Check if the prediction is correct
    if predicted_label == y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / len(x_test)
print(f'Accuracy (Quantized Model on Edge TPU): {accuracy * 100:.2f}%')
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

phase_name = 'Memory utilization'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
