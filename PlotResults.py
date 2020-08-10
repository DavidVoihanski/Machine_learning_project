import keras
from keras import Model
from matplotlib import pyplot as plt
from keras.datasets import cifar10
from keras.utils import plot_model


def plot(model):
    """
    Saves png graph of model
    :param model:
    """
    plot_model(model,
               to_file="/home/david/Desktop/Year3Sem2/Machine Learning/Project/model.png",
               show_shapes=True,
               show_layer_names=True,
               rankdir="TB",
               expand_nested=False,
               dpi=96)


def restore_model():
    """
    Restores the model
    :return: the model
    """
    path_to_model_save = "/home/david/Desktop/Year3Sem2/Machine Learning/Project/cnn_detector/cifar_model/my_model"
    model = keras.models.load_model(path_to_model_save)
    return model


if __name__ == "__main__":
    path_to_save = "/home/david/Desktop/Year3Sem2/Machine Learning/Project/filter_results/"
    model = restore_model()
    plot(model)
    layer_name = 'conv2d_1'
    encoder_layer = Model(inputs=model.input,
                          outputs=model.get_layer(layer_name).output)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    encoded = encoder_layer([x_train[1967:1968]])[0]
    plt.hot()
    plt.imshow(x_train[1967])
    plt.show()
    plt.imsave(path_to_save + "original_" + ".png", x_train[1967])
    for i in range(0, 257):
        # plt.imshow(encoded[:, :, i])
        # plt.show()
        abs_path = path_to_save + layer_name + "_" + str(i) + ".png"
        plt.imsave(abs_path, encoded[:, :, i])
