from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import gc
from sklearn import svm
from sklearn import tree
import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.utils import to_categorical
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import time
import pickle
import matplotlib.pyplot as plt
from PIL import Image

"""

"""


def encode(img_list: np.ndarray, encoder_layer) -> np.ndarray:
    img_list_out = []
    for img in img_list:
        img_shape = np.ndarray(shape=[1, 32, 32, 3])
        img_shape[0] = img
        img = encoder_layer([img_shape])[0]
        img_list_out.append(img)
    return img_list_out


def restore_model():
    path_to_model_save = '../cnn_detector/cifar_model/my_model'
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
    print("KNN started...")

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(train_data_x, train_data_y)

    filename = 'saved-models/knn-model-train.sav'
    pickle.dump(classifier, open(filename, 'wb'))

    y_pred = classifier.predict(test_data_x)
    print("KNN accuracy: ", np.mean(y_pred == test_data_y))


    # ------------------


def SVM(train_data_x, train_data_y, test_data_x, test_data_y):
    """
    :param train_data_x:
    :param train_data_y:
    :param test_data_x:
    :param test_data_y:
    :return:
    """
    print("SVM started...")
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(train_data_x, train_data_y)

    filename = 'saved-models/svm-model-train.sav'
    pickle.dump(clf, open(filename, 'wb'))

    pred_y = clf.predict(test_data_x)
    print("SVM accuracy: ", np.mean(pred_y == test_data_y))


def naive_bayes(train_data_x, train_data_y, test_data_x, test_data_y):
    """
    :param train_data_x:
    :param train_data_y:
    :param test_data_x:
    :param test_data_y:
    :return:
    """
    print("Naive-Bayes started...")
    clf = MultinomialNB()
    clf.fit(train_data_x, train_data_y)

    filename = 'saved-models/nb-model-train.sav'
    pickle.dump(clf, open(filename, 'wb'))

    pred_y = clf.predict(test_data_x)
    print("Naive-Bayes accuracy: ", np.mean(pred_y == test_data_y))

    tsne = TSNE().fit_transform(pred_y)
    tx, ty = tsne[:, 0], tsne[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

    width = 4000
    height = 3000
    max_dim = 100
    full_image = Image.new('RGB', (width, height))
    for idx, x in enumerate(x_test):
        tile = Image.fromarray(np.uint8(x * 255))
        rs = max(1, tile.width / max_dim, tile.height / max_dim)
        tile = tile.resize((int(tile.width / rs),
                            int(tile.height / rs)),
                           Image.ANTIALIAS)
        full_image.paste(tile, (int((width - max_dim) * tx[idx]),
                                int((height - max_dim) * ty[idx])))
    plt.figure(figsize=(16, 12))
    plt.imshow(full_image)


def dec_tree(train_data_x, train_data_y, test_data_x, test_data_y):
    print("Dec' tree started...")
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data_x, train_data_y)

    filename = 'saved-models/d-tree-model-train.sav'
    pickle.dump(clf, open(filename, 'wb'))

    pred_y = clf.predict(test_data_x)
    print("Dec' tree accuracy: ", np.mean(pred_y == test_data_y))






def test_using_storage(saved_model, test_data_x, test_data_y):
    print(saved_model.split('/')[1], "started")
    loaded_model = pickle.load(open(saved_model, 'rb'))
    score = loaded_model.score(test_data_x, test_data_y)
    print("Accuracy: {0:.2f} %".format(100 * score))

    # pred_y = loaded_model.score(test_data_x,test_data_x)

    # print(saved_model.split('/')[1], ": ", np.mean(pred_y != test_data_y))


if __name__ == '__main__':
    start_time = time.time()

    # get the features vector (250 features)
    model = restore_model()
    layer_name = 'dense_2'
    encoder_layer = Model(inputs=model.input,
                          outputs=model.get_layer(layer_name).output)

    # load and prepare the data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)

    step1 = input("type t to train the whole model, or any for testing using storage\n")
    print()
    x_test = encode(x_test, encoder_layer)


    if step1 == 't':
        x_train = encode(x_train, encoder_layer)

        # run the algorithms
        # KNN(x_train, y_train_one_hot, x_test, y_test_one_hot)
        # SVM(x_train, y_train.flatten(), x_test, y_test.flatten())
        naive_bayes(x_train, y_train.flatten(), x_test, y_test.flatten())
        # dec_tree(x_train, y_train_one_hot, x_test, y_test_one_hot)
        gc.collect()
    else:

        # test_using_storage('saved-models/nb-model-train.sav', x_test, y_test)
        # test_using_storage('saved-models/knn-model-train.sav', x_test, y_test)

        test_using_storage('saved-models/svm-model-train.sav', x_test, y_test)

        # test_using_storage('saved-models/d-tree-model-train.sav', x_test, y_test)

    print("--- %s minutes ---" % ((time.time() - start_time) / 60))
