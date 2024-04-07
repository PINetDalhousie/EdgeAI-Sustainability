import os
import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# Quantize the data to 16-bit precision
train_images_quantized = tf.cast(train_images, tf.float16)
test_images_quantized = tf.cast(test_images, tf.float16)

# Define a simple sequential model with quantization-aware training
def create_model():
    model = tf.keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights during training
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the quantization-aware model with the new callback
model.fit(train_images_quantized,
          train_labels,
          epochs=10,
          validation_data=(test_images_quantized, test_labels),
          callbacks=[cp_callback])  # Pass callback to training

# Create and train a new model instance.
model = create_model()
model.fit(train_images_quantized, train_labels, epochs=5)

# Save the entire quantized model to an HDF5 file with 16-bit precision.
model.save('neural_network_model_float16.h5', include_optimizer=False)
