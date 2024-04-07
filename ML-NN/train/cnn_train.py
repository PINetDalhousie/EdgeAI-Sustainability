import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values to [0, 1]
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Reshape the data to add a channel dimension for Conv2D
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Build a CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Save the trained model
model.save('cnn_model.h5')

# Convert the model to a TensorFlow Lite quantized model with FP16
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
quantized_tflite_model = converter.convert()

# Save the quantized model to a file
with open('quantized_cnn_model_fp16.tflite', 'wb') as f:
    f.write(quantized_tflite_model)

# Load the quantized model
interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Evaluate the quantized model on the test data
quantized_accuracy = 0
for i in range(len(x_test)):
    interpreter.set_tensor(input_details[0]['index'], x_test[i:i+1].astype('float32'))  # Ensure input is float32
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    quantized_accuracy += (y_test[i:i+1].argmax(axis=1) == output.argmax(axis=1)).mean()

quantized_accuracy /= len(x_test)
print(f'Quantized Model Accuracy: {quantized_accuracy * 100:.2f}%')
