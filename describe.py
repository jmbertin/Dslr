import pandas as pd
import sys
from utils import count, mean, std, mini, quantile, median, maxi, missing_values, variance, range_value

sys.tracebacklimit = 0


def main():
    """
    Main function that reads data from a CSV file, performs basic descriptive statistics on it,
    and prints the results in a structured format.
    The function reads data from a CSV file whose path should be passed as a command-line argument.
    The function will terminate the program if no file path is provided, if the file cannot be read,
    or if the dataset read from the file is empty.
    Args: sys.argv[1] (str): The first command-line argument is supposed to be a path to the CSV file to read.
    Raises: AssertionError: If no file path is provided or if the dataset read from the file is empty.
            SystemExit: If the file cannot be read.
    Returns: None. However, the statistics of the data will be printed as output.
    """
    assert len(sys.argv) == 2, "Invalid arguments"
    try:
        features = pd.read_csv(sys.argv[1])
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    assert features.empty == False, "Dataset is empty"

    feature_list = list(features.columns)[6:]

    informations: dict = {}
    for value in feature_list:
        informations[value] = {
            "Count: ": count(features[value]),
            "Mean: ": mean(features[value]),
            "Std: ": std(features[value]),
            "Min: ": mini(features[value]),
            "25%: ": quantile(features[value], 0.25),
            "50%: ": median(features[value]),
            "75%: ": quantile(features[value], 0.75),
            "Max: ": maxi(features[value]),
            "Variance: ": variance(features[value]), # Bonus
            "Missing: ": missing_values(features[value]), # Bonus
            "Range: ": range_value(features[value]), # Bonus
            "10%: ": quantile(features[value], 0.1), # Bonus
            "90%: ": quantile(features[value], 0.9) # Bonus
        }

    column_width = 12

    print("           ", end="")
    for i, value in enumerate(feature_list):
        value_header = value if len(value) <= column_width else value[:column_width-1] + '.'
        if i == len(feature_list) - 1:
            print(value_header, end="")
        else:
            print("{:<{width}}".format(value_header, width=column_width), end=" ")
    print()

    for key in ["Count: ", "Mean: ", "Std: ", "Min: ", "25%: ", "50%: ", "75%: ", "Max: ", "Variance: ", "Missing: ", "Range: ", "10%: ", "90%: "]:
        key_header = key if len(key) <= column_width else key[:column_width-1] + '.'
        print("{:<{width}}".format(key_header, width=10), end=" ")
        for i, value in enumerate(feature_list):
            value_formatted = round(informations[value][key], 5)
            value_output = str(value_formatted) if len(str(value_formatted)) <= column_width else str(value_formatted)[:column_width-1] + '.'
            if i == len(feature_list) - 1:
                print(value_output, end="")
            else:
                print("{:<{width}}".format(value_output, width=column_width), end=" ")
        print()

if __name__ == '__main__':
    main()
