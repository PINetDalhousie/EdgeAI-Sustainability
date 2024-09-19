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
# Load the TensorFlow Lite model.
interpreter = tf.lite.Interpreter(model_path="tinybert_finetuned.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'Load data'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

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

# Perform any post-processing if necessary
# For example, if you have a classification task, you may need to apply softmax
# output_data = tf.nn.softmax(output_data, axis=-1)

# Print or use the output
print(output_data)

print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'memory_util'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")