import pandas as pd
import matplotlib.pyplot as plt
import sys
from textwrap import wrap

sys.tracebacklimit = 0


def main():
    """
    The main function for the script, designed to read a CSV file of features,
    segregate it based on 'Hogwarts House' and scatter plot the data for analysis.
    This function will terminate the program if file is not found, file is not provided,
    or if the dataset read from the file is empty.
    The data is categorized into 4 Hogwarts houses: Gryffindor, Ravenclaw, Slytherin, and Hufflepuff.
    It then plots a scatter graph where data points of each category are represented with different colors.
    Args: sys.argv[1] (str): The first command-line argument is supposed to be a path to the CSV file to read.
    Raises: AssertionError: If no file is provided or if the dataset read from the file is empty.
            SystemExit: If the file is not found or cannot be read.
    Returns: None. However, a scatter plot will be shown as output.
    """
    assert len(sys.argv) == 2, "Invalid arguments"
    try:
        features = pd.read_csv(sys.argv[1])
    except:
        print("Error: file not found")
        sys.exit(1)

    assert features.empty == False, "Dataset is empty"

    feature_list = list(features.columns)[6:]

    gryffindor = features[features["Hogwarts House"] == "Gryffindor"]
    ravenclaw = features[features["Hogwarts House"] == "Ravenclaw"]
    slytherin = features[features["Hogwarts House"] == "Slytherin"]
    hufflepuff = features[features["Hogwarts House"] == "Hufflepuff"]

    labels = ['\n'.join(wrap(l, 20)) for l in feature_list]

    plt.scatter(gryffindor[feature_list[1]], gryffindor[feature_list[3]], alpha=0.5, color="red" , label="Gryffindor")
    plt.scatter(ravenclaw[feature_list[1]], ravenclaw[feature_list[3]], alpha=0.5, color="blue", label="Ravenclaw")
    plt.scatter(slytherin[feature_list[1]], slytherin[feature_list[3]], alpha=0.5, color="green", label="Slytherin")
    plt.scatter(hufflepuff[feature_list[1]], hufflepuff[feature_list[3]], alpha=0.5, color="yellow", label="Hufflepuff")
    plt.legend(loc='upper right')
    plt.xlabel(labels[3])
    plt.ylabel(labels[1])
    plt.show()


if __name__ == "__main__":
    main()
