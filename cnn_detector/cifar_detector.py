"""
python code:
https://www.thepythoncode.com/article/use-transfer-learning-for-image-flower-classification-keras-python

dog and cat classification:
https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/

Analytics vidhya - img to vector
https://www.analyticsvidhya.com/blog/2019/08/3-techniques-extract-features-from-image-data-machine-learning-python/


http://www.marekrei.com/blog/transforming-images-to-feature-vectors/

https://www.youtube.com/watch?v=iGWbqhdjf2s
"""
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import to_categorical, plot_model
from keras.datasets import cifar10

plt.style.use('fivethirtyeight')


def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def train():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)

    # Normalize
    x_train = x_train / 255
    x_test = x_test / 255

    # create model

    model = Sequential()

    # first layer
    model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(32, 32, 3), padding="SAME"))
    model.add(Conv2D(32, (5, 5), activation='relu', padding="SAME"))
    # pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # second layer
    model.add(Conv2D(64, (5, 5), activation='relu', padding="SAME"))
    model.add(Conv2D(128, (5, 5), activation='relu', padding="SAME"))
    # second pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # third conv layer
    model.add(Conv2D(256, (5, 5), activation='relu', padding="SAME"))
    # third pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # flat layer
    model.add(Flatten())

    # add a layer with 1000 neurons
    model.add(Dense(1000, activation='relu'))

    # dropout layer
    model.add(Dropout(0.5))

    # add a layer with 500 neurons
    model.add(Dense(500, activation='relu'))

    # dropout layer
    model.add(Dropout(0.5))

    # add a layer with 250 neurons
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.5))

    # add a dense layer with 500 neurons
    model.add(Dense(500, activation='relu'))

    # add a layer with 10 neurns
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    hist = model.fit(x_train, y_train_one_hot, batch_size=256, epochs=20, validation_split=0.2)

    save_path = "/home/david/Desktop/Year3Sem2/Machine Learning/Project/cnn_detector/cifar_model/my_model"
    model.save(filepath=save_path)

    model.evaluate(x_test, y_test_one_hot)[1]

    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Accuracy')
    plt.ylabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model Los')
    plt.xlabel('Loss')
    plt.ylabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()


if __name__ == '__main__':
    train()
