import os
from PIL import Image
import numpy as np

num_of_classes = 10
epochs = 500
add_ones = True

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

    return np_images, labels


def train_perceptron(initial_weight_vector, eita, training_data, labels):
    ws = []
    for cls in range(num_of_classes):
        if add_ones:
            weight_vector = np.append(initial_weight_vector, 1)
        else:
            weight_vector = initial_weight_vector
        for i in range(epochs):
            no_error = True
            for j in range (len(labels)):
                data_point = training_data[j]
                label = labels[j]
                predicted = np.dot(np.transpose(weight_vector), data_point)
                t = compute_t(label, cls)

                if t == 1 and predicted >= 0:
                    continue
                elif t == -1 and predicted <= 0:
                    continue
                else:
                    no_error = False
                    weight_vector = compute_new_weight_vector(weight_vector, eita, data_point, t)
            if no_error:
                break

        ws.append(weight_vector)
    return np.array(ws)

def compute_new_weight_vector(weight_vector, eita, data_point, t):
    return np.add(weight_vector, eita * data_point * t)

def compute_t(label, weight_vector_index):
    return 1 if label == weight_vector_index else -1

def main():
    training_data, training_labels = read_images("./Train")
    test_data, test_labels = read_images("./Test")

    # Prepare initial weight vector
    initial_weight_vector = np.array([1])
    # the -2 here is because the last entry w0 should be 1 and the data_points have an extra 1 at the end
    for _ in range(len(training_data[0]) - 2):
        initial_weight_vector = np.append(initial_weight_vector, 0)

    for i in range(10):
        eita = 10 ** (-1 * i)
        weights = train_perceptron(initial_weight_vector, eita, training_data, training_labels) # 10 * 785
        predictions = np.transpose(np.dot(weights, np.transpose(test_data))) # 200 * 10

        write_confusion_matrix(i, predictions, test_labels)

def write_confusion_matrix(power_of_eita, predictions, test_labels):
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

    with open("confusion_" + str(power_of_eita) + ".csv", "w") as f:
        eita = 10 ** (-1 * power_of_eita)
        # f.write(str(power_of_eita) + "\n")

        str_confusion_matrix = "Real Value \\ Predicted Value,"
        str_confusion_matrix += ",".join([str(cls) for cls in range(num_of_classes)]) + "\n"
        for i, row in enumerate(confusion_matrix):
            str_confusion_matrix += str(i) + ","
            for num in row:
                str_confusion_matrix += str(int(num)) + ", "

            str_confusion_matrix += "\n"

        f.write(str_confusion_matrix)

if __name__ == '__main__':
    main()