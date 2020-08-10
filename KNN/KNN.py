from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as et
import os
import gc
from sklearn import svm
from sklearn import tree
import keras
from keras import backend as K
from keras.datasets import cifar10
from keras.models import Model
from keras.utils import to_categorical


def encode(img_list: np.ndarray, encoder_layer) -> np.ndarray:
    """
    :param img_list:
    :param encoder_layer:
    :return:
    """
    # img_list_out = []
    # for img in img_list:
    #     img_shape = np.ndarray(shape=[1, 32, 32, 3])
    #     img_shape[0] = img
    #     img = encoder_layer([img_shape])[0]
    #     img_list_out.append(img)
    img = encoder_layer([img_list])[0]

    return img


def restore_model():
    """
    :return:
    """
    path_to_model_save = "/home/david/Desktop/Year3Sem2/Machine Learning/Project/cnn_detector/cifar_model/my_model"
    model = keras.models.load_model(path_to_model_save)
    return model


def KNN(train_data_x, train_data_y, test_data_x, test_data_y):
    """
    :param train_data_x:
    :param train_data_y:
    :param test_data_x:
    :param test_data_y:
    :return:
    """
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(train_data_x, train_data_y)
    y_pred = classifier.predict(test_data_x)
    print("KNN error: ", np.mean(y_pred != test_data_y))


def SVM(train_data_x, train_data_y, test_data_x, test_data_y):
    """
    :param train_data_x:
    :param train_data_y:
    :param test_data_x:
    :param test_data_y:
    :return:
    """
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(train_data_x, train_data_y)
    pred_y = clf.predict(test_data_x)
    print("SVM error: ", np.mean(pred_y != test_data_y))


def dec_tree(train_data_x, train_data_y, test_data_x, test_data_y):
    """

    :param train_data_x:
    :param train_data_y:
    :param test_data_x:
    :param test_data_y:
    :return:
    """
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data_x, train_data_y)
    pred_y = clf.predict(test_data_x)
    print("Dec tree Error: ", np.mean(pred_y != test_data_y))


if __name__ == '__main__':
    model = restore_model()
    layer_name = 'dense_2'
    encoder_layer = Model(inputs=model.input,
                          outputs=model.get_layer(layer_name).output)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255
    x_test = x_test / 255
    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)
    print(y_train_one_hot.shape)
    print("Started encoding train")
    x_train = encode(x_train, encoder_layer)
    print("Started encoding test")
    x_test = encode(x_test, encoder_layer)
    print("Started KNN")
    KNN(x_train, y_train_one_hot, x_test, y_test_one_hot)
    # SVM(x_train, y_train, x_test, y_test)
    print("Started decision tree")
    dec_tree(x_train, y_train_one_hot, x_test, y_test_one_hot)
    gc.collect()
