import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Resize images to meet the minimum input size requirement of MobileNet
x_train_resized = tf.image.resize(x_train[..., tf.newaxis], (32, 32))
x_test_resized = tf.image.resize(x_test[..., tf.newaxis], (32, 32))

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define MobileNet model architecture
def create_mobilenet_model():
    input_tensor = Input(shape=(32, 32, 1))
    base_model = tf.keras.applications.MobileNet(input_shape=(32, 32, 1), include_top=False, weights=None)
    x = base_model(input_tensor)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    output_tensor = Dense(10, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

# Create MobileNet model
model = create_mobilenet_model()

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Define a checkpoint callback
checkpoint = ModelCheckpoint('mobilenet_mnist.h5', monitor='val_loss', save_best_only=True, verbose=1)

# Train the model
model.fit(x_train_resized, y_train, validation_data=(x_test_resized, y_test), epochs=5, batch_size=32, callbacks=[checkpoint])

# Save the trained model
model.save('mobilenet_mnist.h5')
print("Model saved successfully.")

# Load the saved model
saved_model_path = 'mobilenet_mnist.h5'
# saved_model_path = 'mobilenet_imagenet.h5'
model = tf.keras.models.load_model(saved_model_path)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
tflite_model_path = 'mobilenet_mnist.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print("Model converted and saved as:", tflite_model_path)