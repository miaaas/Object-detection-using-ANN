import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the layers for your neural network
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, activation=None):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        if self.activation is not None:
            self.output = self.activation.forward(self.output)

# ReLU activation function
class Activation_ReLU:
    def forward(self, inputs):
        return np.maximum(0, inputs)

# Softmax activation function
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

# Step 1: Data Preparation
data_dir = "Desktop/AI"
folders = ["5star-7star.v21.clip", "mofang.v31.yolov8", "real-ball.v2.v1i.yolov8", "TeamPrompt-WorldChampionShip-BIG.v8i.yolov8"]
class_names = ['star', 'cube', 'pyramid', 'ball']
image_size = (48, 48)

def load_data():
    images = []
    labels = []
    for i, folder in enumerate(folders):
        folder_path = os.path.join(data_dir, folder)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, image_size)
            images.append(image)
            labels.append(i)
    return np.array(images), np.array(labels)

images, labels = load_data()
images = images.reshape(-1, 48, 48, 1)  # Add channel dimension
labels = tf.keras.utils.to_categorical(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Step 2: Model Architecture
model = models.Sequential([
    layers.Flatten(input_shape=(48, 48, 1)),  # Input layer
    Layer_Dense(48*48, 256, activation=Activation_ReLU()),  # Hidden layer 1
    Layer_Dense(256, 128, activation=Activation_ReLU()),   # Hidden layer 2
    Layer_Dense(128, 64, activation=Activation_ReLU()),    # Hidden layer 3
    Layer_Dense(64, 4, activation=Activation_Softmax())    # Output layer
])

# Step 3: Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 4: Training
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Step 5: Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)
