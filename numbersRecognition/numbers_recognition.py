import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
import cv2
import numpy as np
import os

def read_img(file):
    """Read the image, invert colors, resize and prepare for prediction"""
    image = 255 - cv2.imread(file, 0)
    image = cv2.resize(image, (28, 28))
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1)
    image = image.astype('float32')
    return image / 255.0

def load_data():
    """Load and preprocess MNIST dataset - handwritten digits"""
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape((60000, 28, 28, 1))
    X_test = X_test.reshape((10000, 28, 28, 1))
    X_train = X_train.astype('float') / 255
    X_test = X_test.astype('float') / 255
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    return (X_train, Y_train), (X_test, Y_test)

def build_model():
    """Built the CNN model"""
    cnn = Sequential()
    cnn.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(units=128, activation='relu'))
    cnn.add(Dense(units=10, activation='softmax'))
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return cnn

def main():
    # Load and preprocess the data
    (X_train, Y_train), _ = load_data()

    # Build the model
    cnn = build_model()

    # Train the model
    cnn.fit(X_train, Y_train, epochs=5, batch_size=64, validation_split=0.1)

    # Predict labels for image in the current directory
    for file_name in os.listdir():
        if file_name.endswith('.png'):
            image = read_img(file_name)
            label = cnn.predict(image)
            print("Number {} prediction = {}".format(file_name[:-4], np.argmax(label[0])))
        else:
            continue


if __name__ == "__main__":
    main()
