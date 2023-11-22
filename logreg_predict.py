import numpy as np
import pandas as pd
import sys

sys.tracebacklimit = 0


def normalize(X):
    """
    Normalizes the features in the given dataset.
    Normalization is done based on the mean and standard deviation of each feature in the dataset.
    Args: X (np.ndarray): The data to be normalized.
    Returns: X (np.ndarray): The normalized dataset.
    """
    mean = np.array([])
    std = np.array([])

    for i in range(X.shape[1]):
        mean = np.append(mean, X[:, i].mean())
        std = np.append(std, X[:, i].std())

    assert not (0 in std), "Error in dataset, std is null"

    X = (X - mean) / std
    return X

def main():
    """
    The main function for the script.
    It loads a dataset and weights from files, pre-processes the data, calculates the scores for each class
    using the loaded weights, and then writes the predictions to a CSV file.
    Args: sys.argv[1] (str): The first command-line argument is supposed to be a path to the CSV file to read.
          sys.argv[2] (str): The second command-line argument is supposed to be a path to the .npy file containing the weights.
    Raises: AssertionError: If not enough command-line arguments are provided.
            SystemExit: If the files cannot be found or read.
    Returns: None. However, a CSV file 'houses.csv' with the predicted house for each student will be written as output.
    """
    assert len(sys.argv) == 3, "Invalid arguments"
    try:
        weights = np.load(sys.argv[2])
    except:
        print("Error, wrong file for the weights")
        sys.exit(1)

    try:
        dataset = pd.read_csv(sys.argv[1])
    except Exception as e:
        print("Error, wrong file for the dataset")
        sys.exit(1)

    dataset = dataset.drop(columns=['First Name', 'Last Name', 'Birthday', 'Best Hand'])

    y_values = dataset.values[:, 1]
    x_values = np.array(dataset.values[:, [4, 5, 6, 7, 14]], dtype=float)
    x_values = np.nan_to_num(x_values)

    X = normalize(x_values)

    scores = np.dot(X, weights)
    predicted_indices_test = np.argmax(scores, axis=1)

    house_names = ["Slytherin", "Gryffindor", "Hufflepuff", "Ravenclaw"]
    predicted_labels_test = [house_names[idx] for idx in predicted_indices_test]

    df = pd.DataFrame({'Index': range(len(predicted_labels_test)), 'Hogwarts House': predicted_labels_test})
    df.to_csv('houses.csv', index=False)


if __name__ == '__main__':
    main()
