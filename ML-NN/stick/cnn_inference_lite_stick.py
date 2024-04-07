import numpy as np
import time
from openvino.inference_engine import IECore

# Load the MNIST test dataset
from tensorflow.keras.datasets import mnist
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)
y_test = tf.keras.utils.to_categorical(y_test)

# Initialize accuracy counter
quantized_accuracy = 0

# Load model
phase_name = 'Load model'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Load the quantized TensorFlow Lite model
model_xml = 'cnn_lite.xml'
model_bin = 'cnn_lite.bin'

# Initialize the Inference Engine Core
ie = IECore()

# Load the network
net = ie.read_network(model=model_xml, weights=model_bin)

# Load the network to the plugin
exec_net = ie.load_network(network=net, device_name='MYRIAD')

print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Perform inference
phase_name = 'Inference'
print(f"{phase_name} - Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

for i in range(len(x_test)):
    input_data = np.array(x_test[i:i + 1], dtype=np.float32)

    # Start an asynchronous request
    request_handle = exec_net.start_async(request_id=0, inputs={'input': input_data})

    # Wait for the request to be complete
    request_handle.wait()

    # Get the inference result
    output = request_handle.outputs['output']

    quantized_accuracy += np.argmax(y_test[i:i + 1]) == np.argmax(output)

quantized_accuracy /= len(x_test)
print(f'Quantized Model Accuracy (FP16): {quantized_accuracy * 100:.2f}%')
print(f"{phase_name} - End: {time.strftime('%Y-%m-%d %H:%M:%S')}")