import cv2
import numpy as np
import time
from openvino.inference_engine import IECore

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
model_path = "tinybert_fp16.tflite"  # Path to your saved TensorFlow Lite TinyBERT model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Load OpenVINO IECore
ie = IECore()

# Load OpenVINO IR model
ir_model_path = "tinybert.xml"  # Path to your OpenVINO IR model
net = ie.read_network(model=ir_model_path, weights=ir_model_path.replace('xml', 'bin'))
exec_net = ie.load_network(network=net, device_name="MYRIAD")  # MYRIAD is the device for Intel NCS
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'Load data'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Prepare input data
# Example input data (tokenized and converted to input_ids and attention_masks)
input_ids = [101, 1045, 2342, 2003, 2025, 1037, 2204, 1012, 102, 2003, 2025, 1037, 2204, 1012, 102]  # Example input_ids
attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Example attention_mask

# Resize input tensor if necessary
input_shape = input_details[0]['shape']
input_data = np.array(input_ids, dtype=np.int32)
input_data = np.resize(input_data, input_shape)
input_mask = np.array(attention_mask, dtype=np.int32)
input_mask = np.resize(input_mask, input_shape)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Set input tensor values
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.set_tensor(input_details[1]['index'], input_mask)

# Run inference
interpreter.invoke()

# Get output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'memory_util'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")