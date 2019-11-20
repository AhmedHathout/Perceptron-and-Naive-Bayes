import os
from PIL import Image
import numpy as np
import math

num_of_classes = 10
add_ones = False
num_of_training_images_per_class = 240
num_of_test_images_per_class = 20
total_num_of_pixels = 784
min_variance_value = 0.01

def read_images(folder: str):
    np_images = [[] for filename in os.listdir(folder) if filename.endswith(".jpg")]

    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            index = int(filename.split(".")[0]) - 1
            image = Image.open(os.path.join(folder, filename))
            np_image = np.array(image).flatten()

            if add_ones:
                np_image = np.append(np_image, 1)

            del np_images[index]
            np_images.insert(index, np_image)

    with open(os.path.join(folder, "Labels.txt"), "r") as f:
        labels = [int(label.strip()) for label in f.readlines()]

    return np.array(np_images), labels

def get_mean_and_standard_deviation(data_points):
    split_data_points = np.split(data_points, num_of_classes)
    mean = np.zeros(shape=(num_of_classes, total_num_of_pixels))
    std = np.zeros(shape=(num_of_classes, total_num_of_pixels))

    for i, cls in enumerate(split_data_points):
        mean[i] = cls.mean(axis=0)
        std[i] = cls.std(axis=0)

    return mean, std

def gaussian(input,pi,mean,std):
    if std==0:
        return min_variance_value
    std = max(0.1, std)
    result = 1/(math.sqrt(2*pi)*std)
    result = result*math.exp(-0.5*math.pow(input-mean,2)/math.pow(std,2))
    # if result<min_variance_value:
    #     result = min_variance_value
    return result

def write_confusion_matrix(predictions, test_labels):
    normalisation_matrix = []
    for i in range(len(predictions)):
        normalisation_matrix.append(np.zeros(10).tolist())
        max_index = 0

        for j in range(len(predictions[i])):
            if predictions[i][j] > predictions[i][max_index]:
                max_index = j

        normalisation_matrix[i][max_index] += 1

    confusion_matrix = [[0 for _ in range(num_of_classes)] for _ in range(num_of_classes)]
    for i in range(len(normalisation_matrix)):
        for j in range(len(normalisation_matrix[0])):
            if normalisation_matrix[i][j] == 1:
                confusion_matrix[i // 20][j] += 1
                break

    with open("Confusion-Gauss.csv", "w") as f:

        str_confusion_matrix = "Real Value \\ Predicted Value,"
        str_confusion_matrix += ",".join([str(cls) for cls in range(num_of_classes)]) + "\n"
        for i, row in enumerate(confusion_matrix):
            str_confusion_matrix += str(i) + ","
            for num in row:
                str_confusion_matrix += str(int(num)) + ", "

            str_confusion_matrix += "\n"

        f.write(str_confusion_matrix)

def main():
    training_data, training_labels = read_images("./Train")
    test_data, test_labels = read_images("./Test")
    training_data = np.true_divide(training_data, 255)
    test_data = np.true_divide(test_data, 255)
    pi = 1/num_of_classes
    pi = 3.14

    mean, std = get_mean_and_standard_deviation(training_data)

    predictions = np.zeros(shape=(200, num_of_classes))
    for i, test_point in enumerate(test_data):
        probs = np.zeros(shape=(num_of_classes))
        for j in range(num_of_classes):
            prob = 1#math.pow(10, 300)
            for l in range(total_num_of_pixels):
                prob = prob * gaussian(test_point[l], pi, mean[j][l], std[j][l])

            probs[j] = prob
        predictions[i] = probs
    write_confusion_matrix(predictions, test_labels)

if __name__ == '__main__':
    main()