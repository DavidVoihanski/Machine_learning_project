import os
import random


def pick_test_images():
    path = "/home/david/Desktop/Year3Sem2/Machine Learning/Project/Images/allImages"
    target = "/home/david/Desktop/Year3Sem2/Machine Learning/Project/Images/testImages"
    file_list = os.listdir(path)
    random.shuffle(file_list)
    sampled = file_list[:4140]
    for file in sampled:
        os.rename(path + "/" + file, target + "/" + file)


def move_to_all_images():
    path = "/home/david/Desktop/Year3Sem2/Machine Learning/Project/Annotation"
    target = "/home/david/Desktop/Year3Sem2/Machine Learning/Project/allLabels"
    folder_list = os.listdir(path)
    for folder in folder_list:
        print("Started Folder")
        file_list = os.listdir(path + "/" + folder)
        for file in file_list:
            os.rename(path + "/" + folder + "/" + file, target + "/" + file)


if __name__ == "__main__":
    # pick_test_images()
    move_to_all_images()

