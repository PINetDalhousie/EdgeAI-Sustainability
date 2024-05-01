from openvino.inference_engine import IECore
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_metric
import torch
import time
import psutil

def get_memory_utilization():
    pid = psutil.Process().pid
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    memory_used = memory_info.rss
    memory_used_mb = memory_used / (1024 * 1024)
    return memory_used_mb

phase_name = 'Load model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the saved model and tokenizer
model_name = "./models/phi-2-orange"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Convert the model to ONNX format
model_onnx_path = "./models/phi-2-orange.onnx"
torch.onnx.export(model, (inputs, ), model_onnx_path)

# Load the IR model
ie = IECore()
model_ir_path = "./models/phi-2-orange.xml"
net = ie.read_network(model=model_ir_path)

phase_name = 'Load dataset'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Load the validation dataset
dataset = load_dataset("OpenAssistant", "oasst1")
valid_dataset = dataset["validation"]
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

accuracy = load_metric("accuracy")

phase_name = 'inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Perform inference on the validation dataset and calculate accuracy
for example in valid_dataset:
    # Encode the inputs and labels
    inputs = tokenizer(example["input_text"], return_tensors="pt")
    labels = torch.tensor([example["label"]])

    # Perform inference
    exec_net = ie.load_network(network=net, device_name="MYRIAD")
    with torch.no_grad():
        outputs = exec_net.infer(inputs=inputs)

    # Compute the accuracy
    logits = outputs['logits']
    predictions = torch.argmax(logits, dim=-1)
    accuracy.add_batch(predictions=predictions, references=labels)
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Print the accuracy
print(f"Accuracy: {accuracy.compute()['accuracy'] * 100:.2f}%")

phase_name = 'memory_util'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
memory_used_mb = get_memory_utilization()
print(f"Memory utilization: {memory_used_mb:.2f} MB")
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")