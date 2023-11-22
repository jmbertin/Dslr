import pandas as pd
import numpy as np
import sys

sys.tracebacklimit = 0

LEARNING_RATE = 0.2
NUM_ITERATIONS = 1000
MB_SIZE = 10


def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Splits a dataset into training and test sets.
    This function shuffles the dataset and then splits it into training and test sets.
    The test set size is determined by `test_size`, which should be a float between 0.0 and 1.0
    and represents the proportion of the dataset to include in the test split.
    Args: X (np.ndarray): The input data.
          y (np.ndarray): The target values.
          test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
          random_state (int, optional): Seed for the random number generator. Defaults to None.
    Returns: X_train, X_test, y_train, y_test (np.ndarray): The split datasets.
    """
    if random_state is not None:
        np.random.seed(random_state)

    num_samples = len(X)
    indices = np.random.permutation(num_samples)
    num_test_samples = int(test_size * num_samples)

    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def normalize(X_train, X_test):
    """
    Normalizes the features of the training and test sets.
    Normalization is done based on the mean and standard deviation of each feature in the training set.
    Args: X_train (np.ndarray): The training data.
          X_test (np.ndarray): The test data.
    Returns: X_train, X_test (np.ndarray): The normalized datasets.
    """
    mean = np.array([])
    std = np.array([])

    for i in range(X_train.shape[1]):
        mean = np.append(mean, X_train[:, i].mean())
        std = np.append(std, X_train[:, i].std())

    assert not (0 in std), "Error in dataset, std is null"

    X_train = (X_train - mean) / std

    X_test = (X_test - mean) / std
    return X_train, X_test


def softmax(scores):
    """
    Computes the softmax of each set of scores in x.
    Softmax is a function that takes as input a vector of K real numbers, and normalizes it into a probability
    distribution consisting of K probabilities proportional to the exponentials of the input numbers.
    In the context of logistic regression, softmax is used to generate probabilities for the output class labels.
    Args: scores (np.ndarray): Input array.
    Returns: np.ndarray: An array the same shape as x. The result will sum to 1 along the rows.
    """
    exp_scores = np.exp(scores)
    sum_scores = np.sum(exp_scores, axis=1, keepdims=True)

    assert not (0 in sum_scores), "Error in dataset, sum of scores is null"

    result = exp_scores / sum_scores
    return result


def get_weights_std(X_train, y_train):
    """
    Trains a logistic regression model and returns the weights.
    The training is performed using batch gradient descent, an iterative optimization algorithm
    for finding the minimum of a function. Here, that function is the loss function of the logistic regression model.
    The learning rate and the number of iterations are defined by the constants LEARNING_RATE and NUM_ITERATIONS.
    Args: X_train (np.ndarray): The training data.
          y_train (np.ndarray): The training labels.
    Returns: weights (np.ndarray): The weights of the trained logistic regression model.
    """
    classes = ["Slytherin", "Gryffindor", "Hufflepuff", "Ravenclaw"]
    n_classes = 4
    n_samples = X_train.shape[0]

    assert n_samples > 0, "Error in dataset, missing data"

    y_one_hot = np.zeros((n_samples, n_classes))

    for i, class_label in enumerate(classes):
        y_one_hot[:, i] = (y_train == class_label).astype(int)

    n_features = X_train.shape[1]
    weights = np.zeros((n_features, n_classes))

    for iteration in range(NUM_ITERATIONS):
        scores = np.dot(X_train, weights)
        probabilities = softmax(scores)

        loss = -np.mean(y_one_hot * np.log(probabilities))

        gradient = np.dot(X_train.T, (probabilities - y_one_hot)) / n_samples

        weights -= LEARNING_RATE * gradient

    return weights


def get_weights_s(X_train, y_train):
    """
    Trains a logistic regression model and returns the weights.
    The training is performed using stochastic gradient descent, an iterative optimization algorithm
    for finding the minimum of a function. Here, that function is the loss function of the logistic regression model.
    The learning rate and the number of iterations are defined by the constants LEARNING_RATE and NUM_ITERATIONS.
    Args: X_train (np.ndarray): The training data.
          y_train (np.ndarray): The training labels.
    Returns: weights (np.ndarray): The weights of the trained logistic regression model.
    """
    classes = ["Slytherin", "Gryffindor", "Hufflepuff", "Ravenclaw"]
    n_classes = 4
    n_samples = X_train.shape[0]

    assert n_samples > 0, "Error in dataset, missing data"

    y_one_hot = np.zeros((n_samples, n_classes))

    for i, class_label in enumerate(classes):
        y_one_hot[:, i] = (y_train == class_label).astype(int)

    n_features = X_train.shape[1]
    weights = np.zeros((n_features, n_classes))

    for iteration in range(NUM_ITERATIONS):
        for i in range(n_samples):
            xi = X_train[i:i+1]
            yi = y_one_hot[i:i+1]

            scores = np.dot(xi, weights)
            probabilities = softmax(scores)

            loss = -np.mean(yi * np.log(probabilities))

            gradient = np.dot(xi.T, (probabilities - yi))

            weights -= LEARNING_RATE * gradient

    return weights


def get_weights_mb(X_train, y_train, batch_size=MB_SIZE):
    """
    Trains a logistic regression model and returns the weights.
    The training is performed using mini-batch gradient descent, an iterative optimization algorithm
    for finding the minimum of a function. Here, that function is the loss function of the logistic regression model.
    The learning rate and the number of iterations are defined by the constants LEARNING_RATE and NUM_ITERATIONS.
    The size of the mini-batch is defined by the `batch_size` argument.
    Args: X_train (np.ndarray): The training data.
          y_train (np.ndarray): The training labels.
          batch_size (int, optional): The number of samples to use in each mini-batch. Defaults to 100.
    Returns: weights (np.ndarray): The weights of the trained logistic regression model.
    """
    classes = ["Slytherin", "Gryffindor", "Hufflepuff", "Ravenclaw"]
    n_classes = 4
    n_samples = X_train.shape[0]

    assert n_samples > 0, "Error in dataset, missing data"

    y_one_hot = np.zeros((n_samples, n_classes))

    for i, class_label in enumerate(classes):
        y_one_hot[:, i] = (y_train == class_label).astype(int)

    n_features = X_train.shape[1]
    weights = np.zeros((n_features, n_classes))

    for iteration in range(NUM_ITERATIONS):
        for i in range(0, n_samples, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_one_hot[i:i+batch_size]

            scores = np.dot(X_batch, weights)
            probabilities = softmax(scores)

            loss = -np.mean(y_batch * np.log(probabilities))

            gradient = np.dot(X_batch.T, (probabilities - y_batch)) / batch_size

            weights -= LEARNING_RATE * gradient

    return weights


def main():
    """
    The main function for the script.
    It loads a dataset, pre-processes the data, trains a logistic regression model,
    and then calculates the accuracy of the model on a test set.
    Args: sys.argv[1] (str): The first command-line argument is supposed to be a path to the CSV file to read.
    Raises: AssertionError: If no file is provided or if the dataset read from the file is empty.
            SystemExit: If the file is not found or cannot be read.
    Returns: None. However, the accuracy of the model will be printed as output.
    """
    assert len(sys.argv) == 2, "Invalid arguments"
    try:
        dataset_train = pd.read_csv(sys.argv[1])
    except:
        print("Error file")
        sys.exit(1)

    assert dataset_train.empty == False, "Dataset is empty"

    dataset_train = dataset_train.drop(columns=['First Name', 'Last Name', 'Birthday', 'Best Hand'])

    y_values = dataset_train.values[:, 1]

    for house in y_values:
        assert isinstance(house, str), "Incomplete dataset, missing house"

    x_values = np.array(dataset_train.values[:, [4, 5, 6, 7, 14]], dtype=float)
    x_values = np.nan_to_num(x_values)

    X_train, X_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.2, random_state=12)

    X_train, X_test = normalize(X_train, X_test)

    print("Choose the gradient descent mode:")
    print("1. Standard Gradient Descent (batch)")
    print("2. Stochastic Gradient Descent")
    print("3. Mini-Batch Gradient Descent")
    mode = input("Enter the mode number: ")

    if mode == "1":
        weights = get_weights_std(X_train, y_train)
    elif mode == "2":
        weights = get_weights_s(X_train, y_train)
    elif mode == "3":
        weights = get_weights_mb(X_train, y_train)
    else:
        print("Unrecognized mode, using Batch Gradient Descent by default.")
        weights = get_weights_std(X_train, y_train)

    scores = np.dot(X_test, weights)

    classes = ["Slytherin", "Gryffindor", "Hufflepuff", "Ravenclaw"]
    predicted_indexes = np.argmax(scores, axis=1)
    predicted_labels = np.array([classes[i] for i in predicted_indexes])

    accuracy = np.mean(predicted_labels == y_test)

    # print("Predicted labels:", predicted_labels)
    # print("True labels:", y_test)
    print("Accuracy:", accuracy)
    np.save("weights.npy", weights)


if __name__ == "__main__":
    main()
