
import pandas as pd
import matplotlib.pyplot as plt
import sys
from textwrap import wrap

from utils import mini, maxi

sys.tracebacklimit = 0


def main():
    """
    The main function for the script, designed to read a CSV file of features,
    segregate it based on 'Hogwarts House' and create pair plots of the features for analysis.
    This function will terminate the program if file is not found, file is not provided,
    the dataset read from the file is empty or missing data for a house.
    The data is categorized into 4 Hogwarts houses: Gryffindor, Ravenclaw, Slytherin, and Hufflepuff.
    It then creates a grid of subplots, where each cell in the grid is either a scatter plot
    (for off-diagonal cells) or a histogram (for diagonal cells) of data points in each category,
    represented with different colors.
    Args: sys.argv[1] (str): The first command-line argument is supposed to be a path to the CSV file to read.
    Raises: AssertionError: If no file is provided, if the dataset read from the file is empty, or if data for any house is missing.
            SystemExit: If the file is not found or cannot be read.
    Returns: None. However, a set of pair plots will be shown as output.
    """
    assert len(sys.argv) == 2, "Invalid arguments"
    try:
        features = pd.read_csv(sys.argv[1])
    except:
        print("Error: file not found")
        sys.exit(1)

    assert features.empty == False, "Dataset is empty"

    feature_list = list(features.columns)[6:]
    nb_features = len(feature_list)

    fig, ax = plt.subplots(nb_features, nb_features, figsize=(10, 10))
    fig.suptitle("Pair plots")

    gryffindor = features[features["Hogwarts House"] == "Gryffindor"]
    ravenclaw = features[features["Hogwarts House"] == "Ravenclaw"]
    slytherin = features[features["Hogwarts House"] == "Slytherin"]
    hufflepuff = features[features["Hogwarts House"] == "Hufflepuff"]

    assert not gryffindor.empty and not ravenclaw.empty and not slytherin.empty and not hufflepuff.empty, "Missing data for a house"


    labels = ['\n'.join(wrap(l, 20)) for l in feature_list]

    for i in range(nb_features):

        for j in range(nb_features):

            if j == 0:
                ax[i, j].set_ylabel(labels[i], rotation=0, labelpad=len(labels[i]) + 30, fontsize=8)
                ax[i, j].tick_params(axis='y', labelsize=6)
                if i != nb_features - 1:
                    ax[i, j].set_xticks([])
                else:
                    ax[i, j].set_xlabel(labels[j], fontsize=8)
                    ax[i, j].tick_params(axis='x', labelsize=6)

            elif i == nb_features - 1:
                ax[i, j].set_xlabel(labels[j], fontsize=8)
                ax[i, j].tick_params(axis='x', labelsize=6)
                ax[i, j].set_yticks([])
            else:
                ax[i, j].set_yticks([])
                ax[i, j].set_xticks([])

            ax[i, j].set_xlim(mini(features[feature_list[j]]), maxi(features[feature_list[j]]))

            if i == j:
                ax[i, j].hist(gryffindor[feature_list[j]], color="red", alpha=0.5, label="Gryffindor")
                ax[i, j].hist(ravenclaw[feature_list[j]], color="blue", alpha=0.5, label="Ravenclaw")
                ax[i, j].hist(slytherin[feature_list[j]], color="green", alpha=0.5, label="Slytherin")
                ax[i, j].hist(hufflepuff[feature_list[j]], color="yellow", alpha=0.5, label="Hufflepuff")
                continue

            ax[i, j].scatter(gryffindor[feature_list[j]], gryffindor[feature_list[i]], color="red", alpha=0.5, label="Gryffindor")
            ax[i, j].scatter(ravenclaw[feature_list[j]], ravenclaw[feature_list[i]], color="blue", alpha=0.5, label="Ravenclaw")
            ax[i, j].scatter(slytherin[feature_list[j]], slytherin[feature_list[i]], color="green", alpha=0.5, label="Slytherin")
            ax[i, j].scatter(hufflepuff[feature_list[j]], hufflepuff[feature_list[i]], color="yellow", alpha=0.5, label="Hufflepuff")

    plt.show()


if __name__ == "__main__":
    main()
