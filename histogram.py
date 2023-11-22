import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

sys.tracebacklimit = 0

def main():
    """
    Main function that generates a histogram showing the distribution of grades for 'Care of Magical Creatures'
    across different houses in Hogwarts.
    The function reads data from a CSV file whose path should be passed as a command-line argument.
    The function will terminate the program if no file path is provided, if the file cannot be read,
    if the dataset read from the file is empty, or if no houses exist in the dataset.
    Args: sys.argv[1] (str): The first command-line argument is supposed to be a path to the CSV file to read.
    Raises: AssertionError: If no file path is provided, if the dataset read from the file is empty, or if no houses exist in the dataset.
            SystemExit: If the file cannot be read.
    Returns: None. However, a histogram plot will be shown as output.
    """
    assert len(sys.argv) == 2, "Invalid arguments"
    try:
        features = pd.read_csv(sys.argv[1])
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    assert features.empty == False, "Dataset is empty"

    subject = 'Care of Magical Creatures'  # Set the subject to be focused on
    houses = features["Hogwarts House"].dropna().unique()
    assert len(houses) > 0, "Dataset error, there is no house to study"

    plt.figure(figsize=(8, 8))
    plt.title("Histogram representing the distribution of grades for '%s'" % subject)
    for house in houses:
        house_features = features[features["Hogwarts House"] == house]
        plt.hist(house_features[subject].dropna(), bins=20, label=house, alpha=0.6)

    plt.xlabel("Grades")
    plt.ylabel("Number of students")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
