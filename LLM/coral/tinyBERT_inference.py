import numpy as np
import psutil
import time
from edgetpu.basic.basic_engine import BasicEngine

def get_memory_utilization():
    pid = psutil.Process().pid
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    memory_used = memory_info.rss
    memory_used_mb = memory_used / (1024 * 1024)
    return memory_used_mb

phase_name = 'Load model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the Edge TPU model
model_path = 'tinybert_edgetpu.tflite'  # Update with your Edge TPU model path
engine = BasicEngine(model_path)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'Load data'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Example input data (tokenized)
input_ids = [101, 1045, 2342, 2003, 2025, 1037, 2204, 1012, 102]  # Example input_ids
attention_mask = [1] * len(input_ids)  # Example attention_mask

# Prepare input data for Edge TPU
input_data = np.array(input_ids, dtype=np.uint8)  # Change dtype as required
attention_mask_data = np.array(attention_mask, dtype=np.uint8)  # Change dtype as required

# Edge TPU typically expects input of specific shapes, so ensure your model's expected shape matches
input_data = np.pad(input_data, (0, max(0, 128 - len(input_data))), 'constant')  # Pad if needed
attention_mask_data = np.pad(attention_mask_data, (0, max(0, 128 - len(attention_mask_data))), 'constant')  # Pad if needed

print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Run inference
results = engine.RunInference([input_data, attention_mask_data])

# The output will be a list of results; process as needed
output_data = np.array(results)
print("Output Data:", output_data)

print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'memory_util'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
