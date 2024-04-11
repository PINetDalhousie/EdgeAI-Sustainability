import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import glue_convert_examples_to_features
from transformers import TFBertForSequenceClassification, BertConfig

# Load pre-trained TinyBERT
model_name = "google/tiny-bert-mini"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name)

# Download and load MRPC dataset from GLUE benchmark
task = "mrpc"
data = tf.keras.utils.get_file(
    "glue_data.zip",
    f"https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2F{task}.zip?alt=media",
    extract=True,
)
data_dir = os.path.join(os.path.dirname(data), f"{task}")
train_dataset = glue_convert_examples_to_features(data_dir, tokenizer, max_length=128, task=task)

# Fine-tune TinyBERT on MRPC dataset
epochs = 3
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

model.fit(train_dataset, epochs=epochs)

# Save the trained model
model.save_pretrained("tinybert_finetuned")

# Convert the TensorFlow model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model("tinybert_finetuned")
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open("tinybert_finetuned.tflite", "wb") as f:
    f.write(tflite_model)