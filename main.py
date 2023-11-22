import os
import sys

def clear_screen():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

def main_menu():
    print("Welcome to the Hogwarts House Prediction and Analysis Program!")
    print("0. Exit")
    print("1. Train Model (mandatory before prediction))")
    print("2. Predict Hogwarts Houses")
    print("3. Compare Results")
    print("4. Generate Histogram")
    print("5. Show Descriptive Statistics")
    print("6. Generate Pair Plot")
    print("7. Generate Scatter Plot")
    choice = input("Enter your choice (0-7): ")
    return choice

def train_model():
    os.system("python3 logreg_train.py dataset_train.csv")

def predict_houses():
    os.system("python3 logreg_predict.py dataset_test.csv weights.npy")

def compare_results():
    os.system("diff houses.csv dataset_truth.csv")

def generate_histogram():
    os.system("python3 histogram.py dataset_train.csv")

def show_descriptive_statistics():
    os.system("python3 describe.py dataset_train.csv")

def generate_pair_plot():
    os.system("python3 pair_plot.py dataset_train.csv")

def generate_scatter_plot():
    os.system("python3 scatter_plot.py dataset_train.csv")

def main():
    while True:
        choice = main_menu()
        if choice == "1":
            train_model()
        elif choice == "2":
            predict_houses()
        elif choice == "3":
            compare_results()
        elif choice == "4":
            generate_histogram()
        elif choice == "5":
            show_descriptive_statistics()
        elif choice == "6":
            generate_pair_plot()
        elif choice == "7":
            generate_scatter_plot()
        elif choice == "0":
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

        input("Press any key to return to the menu...")
        clear_screen()

if __name__ == "__main__":
    main()
