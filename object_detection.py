import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import tensorflow as tf
from tensorflow.keras import layers, models

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
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')
])

# Step 3: Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 4: Training
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Step 5: Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
predictions = model.predict(X_test)
precision = precision_score(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1), average='macro')

print("Test Accuracy:", test_accuracy)
print("Precision:", precision)
print("Test Loss:", test_loss)
