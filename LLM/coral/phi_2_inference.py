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
model_path = 'phi2_edgetpu.tflite'  # Path to your converted Edge TPU model
engine = BasicEngine(model_path)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'Load dataset'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the validation dataset
dataset = load_dataset("OpenAssistant", "oasst1")
valid_dataset = dataset["validation"]
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

phase_name = 'inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Perform inference on the validation dataset
for example in valid_dataset:
    # Prepare input data (tokenized and encoded)
    input_text = example["input_text"]
    # Tokenization logic should match what was used during model training
    input_ids = tokenizer.encode(input_text, return_tensors='np')

    # Resize input tensor if necessary
    input_shape = engine.get_input_tensor_shape(0)
    input_data = np.pad(input_ids, (0, max(0, input_shape[1] - len(input_ids))), 'constant')

    # Run inference
    results = engine.RunInference(input_data)

    # Post-process the results
    predictions = np.argmax(results[0])  # Adjust based on your model's output structure

    # Compare predictions with actual labels for accuracy
    # Replace this with your logic to calculate accuracy based on model outputs
    # Example: accuracy.add_batch(predictions=predictions, references=example["label"])

print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Print the accuracy (implement logic to compute this based on predictions)
# print(f"Accuracy: {accuracy.compute()['accuracy'] * 100:.2f}%")

phase_name = 'memory_util'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
