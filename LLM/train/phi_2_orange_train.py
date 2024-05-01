from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load pre-trained model and tokenizer
model_name = "rhysjones/phi-2-orange"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the dataset
dataset = load_dataset("OpenAssistant", "oasst1")
train_dataset = dataset["train"]
valid_dataset = dataset["validation"]

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./models/phi-2-orange",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Create the Trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

trainer.train()

# Save the model
trainer.save_model("./models/phi-2-orange")