#import the TensorFlow library, which is a popular open-source machine learning framework, and sets up the necessary components for building a neural network using the Keras API within TensorFlow. 
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten

#This line of code loads the MNIST dataset using the Keras library. MNIST is a well-known dataset in the field of machine learning, particularly for testing image classification algorithms. The dataset consists of 28x28 pixel grayscale images of handwritten digits (0 through 9).
(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()

#The variable X_train typically contains the training images from the MNIST dataset. It is a NumPy array with a shape of (num_samples, 28, 28), where num_samples is the number of training examples, and each training example is a 28x28 pixel grayscale image.
X_train

#X_train.shape would output a tuple representing the dimensions of the array. For example, if you execute:
print(X_train.shape)
#And the output is (60000, 28, 28), it means there are 60,000 training images, and each image has a size of 28 pixels by 28 pixels. The specific numbers might vary depending on the size of the dataset used.

# Adjust the index (X_train[index]) to visualize other training images.
print(X_train[0])

#The imshow function is used to display the image.
import matplotlib.pyplot as plt
plt.imshow(X_train[0])

#The code X_train = X_train/255 and X_test = X_test/255 normalizes the pixel values in the training and testing datasets of the MNIST dataset. This is a common preprocessing step in machine learning, especially when working with image data.
X_train = X_train/255
X_test = X_test/255
#If you print X_train[0] after normalizing, you will see the pixel values of the first training image in the MNIST dataset. Since you've normalized the pixel values to be in the range [0, 1], each value in X_train[0] will be between 0 and 1.

#creating a simple neural network model using the Keras Sequential API.
model = Sequential()
model.add(Flatten(input_shape=(28,28))) #Flatten layer: Flattens the input images.
odel.add(Dense(128,activation = 'relu')) #Dense layer with 128 neurons and ReLU activation: Hidden layer introducing non-linearity.
model.add(Dense(10, activation = 'softmax')) #Dense layer with 10 neurons and softmax activation: Output layer for digit classification.

#The model.summary() method provides a summary of the architecture and parameters of the neural network model. It displays information such as the layer types, output shapes, and the number of trainable parameters.
model.summary()

#The model.compile() line sets up the rules and tools for the model to learn from the data. It tells the model how to measure its mistakes (loss function), how to correct them (optimizer), and what to keep track of (metrics) while learning.
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

#this line of code is like telling the model, "Look at these pictures and their labels. Learn from them by adjusting yourself 25 times, and let me know how well you're doing, especially on pictures you haven't seen before." The history variable helps you see how the learning process is going.
history = model.fit(X_train, y_train, epochs= 25, validation_split= 0.2)

#After running this line of code, you have the predicted probabilities (y_prob), and you can use them to analyze how confident the model is in its predictions for each test image.
y_prob = model.predict(X_test)

#y_pred holds the predicted labels based on the highest probability for each image, allowing you to compare the model's predictions with the actual labels to evaluate its performance.
y_pred = y_prob.argmax(axis = 1)

#The code snippet accuracy_score(y_test, y_pred) is using the accuracy_score function from the sklearn.metrics module to calculate the accuracy of the model's predictions on the test dataset.
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred )

#The line plt.imshow(X_test[1]) is using Matplotlib to display the image of a handwritten digit from the test dataset
plt.imshow(X_test[1])

#making a prediction using the trained model for a single test image and printing the predicted class label. 
print(model.predict(X_test[1].reshape(1,28,28)).argmax(axis=1))
