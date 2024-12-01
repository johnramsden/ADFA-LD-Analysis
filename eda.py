import pandas as pd

def eda():
    train_d = pd.read_csv("data/train_data.csv")
    test_d = pd.read_csv("data/test_data.csv")

    print("Training set")

    print(train_d.head())
    print(train_d.describe())
    print(train_d.shape)
    print(train_d["label"].value_counts())

    print("Testing set")

    print(test_d.head())
    print(test_d.describe())
    print(test_d.shape)
    print(test_d["label"].value_counts())

if __name__ == '__main__':
    eda()