from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_metric
import torch
import time
import psutil
import torch2trt
import numpy as np

def get_memory_utilization():
    pid = psutil.Process().pid
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    memory_used = memory_info.rss
    memory_used_mb = memory_used / (1024 * 1024)
    return memory_used_mb

# Load the saved model and tokenizer
model_name = "./models/phi-2-orange"
print(f"Load model - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Load model - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Transform the model to TensorRT
print(f"Conversion to TensorRT - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
model_trt = torch2trt.torch2trt(model, [torch.zeros(1, 1).cuda()])
print(f"Conversion to TensorRT - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Load the validation dataset
print(f"Load dataset - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
dataset = load_dataset("OpenAssistant", "oasst1")
valid_dataset = dataset["validation"]
accuracy = load_metric("accuracy")
print(f"Load dataset - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Perform inference on the validation dataset and calculate accuracy
print(f"Inference - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
for example in valid_dataset:
    # Encode the inputs and labels
    inputs = tokenizer(example["input_text"], return_tensors="pt")
    labels = torch.tensor([example["label"]])

    # Move inputs and labels to the same device as the model
    inputs = {name: tensor.to(model_trt.device) for name, tensor in inputs.items()}
    labels = labels.to(model_trt.device)

    # Perform inference
    with torch.no_grad():
        outputs = model_trt(**inputs)

    # Compute the accuracy
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    accuracy.add_batch(predictions=predictions, references=labels)
print(f"Inference - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Print the accuracy
print(f"Accuracy: {accuracy.compute()['accuracy'] * 100:.2f}%")

# Log memory utilization
print(f"Memory utilization - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"Memory utilization - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
