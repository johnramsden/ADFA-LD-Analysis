import os
import pandas as pd
from sklearn.model_selection import train_test_split

import lib

def preprocess():
    # Define paths
    training_data_path = "data/ADFA-LD/Training_Data_Master/"
    attack_data_path = "data/ADFA-LD/Attack_Data_Master/"

    # Initialize data storage
    data = []

    # Process normal data
    for filename in os.listdir(training_data_path):
        filepath = os.path.join(training_data_path, filename)
        with open(filepath, "r") as file:
            sequence = file.read().strip()
            data.append({"file_name": filename, "sequence": sequence, "label": "normal"})

    # Process abnormal data
    for root, _, files in os.walk(attack_data_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            with open(filepath, "r") as file:
                sequence = file.read().strip()
                data.append({"file_name": filename, "sequence": sequence, "label": "abnormal"})

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Split data into training and testing sets
    train_data, test_data = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)

    # Save to CSV or other format if needed
    train_data.to_csv("train_data.csv", index=False)
    test_data.to_csv("test_data.csv", index=False)

    print("Training and testing data successfully split and saved.")

def check_unique_vals():
    # Load data from CSV files
    train_d = pd.read_csv("data/train_data.csv")
    test_d = pd.read_csv("data/test_data.csv")

    train_sequences = lib.get_seq(train_d['sequence'])
    test_sequences = lib.get_seq(test_d['sequence'])

    uniq = set()

    for a in train_sequences + test_sequences:
        for v in a:
            uniq.add(v)

    print(uniq)

if __name__ == "__main__":
    pass
    # check_unique_vals()