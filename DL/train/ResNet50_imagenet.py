import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Load the dataset
dataset_name = 'imagenet_v2/matched-frequency'
test_dataset, info = tfds.load(name=dataset_name, split='test', with_info=True, as_supervised=True)

# Preprocess the data
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = info.features['label'].num_classes

def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label

test_dataset = test_dataset.map(preprocess_image).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

# Load ResNet50 model with pre-trained ImageNet weights
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Add custom classifier with ResNet50
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Freeze the layers of ResNet50
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define a checkpoint callback
checkpoint = ModelCheckpoint('resnet50_imagenet_subset.h5', monitor='val_loss', save_best_only=True, verbose=1)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Save the trained model
model.save('resnet50_imagenet.h5')
print("Model saved successfully.")

# Load the saved Keras model
saved_model_path = 'resnet50_imagenet.h5'
# saved_model_path = 'resnet50_mnist.h5'
model = tf.keras.models.load_model(saved_model_path)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
tflite_model_path = 'resnet50_imagenet.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print("Model converted and saved as:", tflite_model_path)
