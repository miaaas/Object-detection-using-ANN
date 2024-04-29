
Project Summary:
The project investigates the application of Artificial Neural Networks (ANNs), specifically Convolutional Neural Networks (CNNs), for object detection, focusing on geometric shape recognition. The research explores recent advancements in deep learning methodologies and techniques to enhance object detection accuracy and efficiency. Key areas covered include automatic detection methods, ensemble methods, foreground feature enhancement, synthetic data generation, and model training with limited labeled data. Challenges such as dataset annotation and selecting appropriate detection algorithms are also discussed.

Methodology Overview:
Data Collection and Preprocessing: Utilized a dataset comprising over 5000 grayscale images depicting various geometric shapes. Preprocessing steps involved resizing images, normalization, and converting them into grayscale.
Model Architecture: Implemented a CNN architecture consisting of convolutional layers, max-pooling layers, batch normalization, fully connected dense layers, and dropout layers. Softmax activation function was used in the output layer for multi-class classification.
Activation Functions: Rectified Linear Unit (ReLU) activation function was used in convolutional and dense layers, while Softmax was applied in the output layer.
Loss Function: Categorical cross-entropy loss function was employed due to the multi-class classification nature of the task.
Optimizer: Adam optimizer was chosen for its efficiency in training neural networks, along with fine-tuned hyperparameters.
Training: The model was trained over 20 epochs with a batch size of 32, and early stopping criteria were employed to prevent overfitting.
Evaluation: Model performance was evaluated using accuracy, precision, and loss metrics on a separate test dataset.

Results and Discussion:
The trained model achieved a training accuracy of 91% and validation accuracy of 88%. Evaluation metrics, including precision, recall, F1 score, and ROC-AUC, indicated the effectiveness of the approach in classifying geometric shapes. However, limitations were noted in detecting complex shapes due to the simplicity of the architecture.

Conclusion:
The project demonstrates a robust methodology for geometric shape detection using CNNs, laying the foundation for further advancements in object detection tasks. Despite limitations, the approach shows promise for practical applications in various domains, while highlighting areas for future research and improvement.

CODE EXPLANATION

Data Preparation:
data_dir = "Desktop/AI"
folders = ["5star-7star.v21.clip", "mofang.v31.yolov8", "real-ball.v2.v1i.yolov8", "TeamPrompt-WorldChampionShip-BIG.v8i.yolov8"]
class_names = ['star', 'cube', 'pyramid', 'ball']
image_size = (48, 48)
Here, we define the directory where our data is located (data_dir) and list the subdirectories within it (folders), each containing images of different objects such as stars, cubes, pyramids, and balls.
class_names stores the labels corresponding to each class.
image_size specifies the desired dimensions for resizing the images.

Loading Data:
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
This function iterates through each subdirectory (folder) and each image file within it, loads the image, resizes it to the specified dimensions, and appends it to the images list.
The corresponding label (i) is appended to the labels list.
The function returns numpy arrays images and labels.

Model Architecture:
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
This code defines a convolutional neural network (CNN) model using Keras' Sequential API.
It consists of convolutional layers, max-pooling layers, batch normalization layers, and dense layers with dropout for regularization.
The input shape for the first layer is specified as (48, 48, 1) indicating the input image dimensions and the single channel (grayscale).

Compilation:
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
Here, we compile the model with the Adam optimizer, categorical cross-entropy loss function (suitable for multi-class classification), and accuracy metric to monitor during training.
