import tensorflow as tf

assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM
from tcn import TCN
import lib
from keras_tuner.tuners import RandomSearch

best_tcn_window = 20
best_lstm_window = 20

def best_window(model: str, train_data, test_data):
    """
    Find the best window size for model (tcn|lstm)
    """
    if model not in ["tcn", "lstm"]:
        print("Invalid model")
        return

    # Prepare sequences and labels
    train_sequences = lib.get_seq(train_data['sequence'])
    train_labels = lib.get_labels(train_data['label'])
    
    test_sequences = lib.get_seq(test_data['sequence'])
    test_labels = lib.get_labels(test_data['label'])
   
    # Test various window lengths
    window_lengths = range(15, 101, 5)
    results = []

    # Replace the model creation logic in the loop with the updated function
    for window_length in window_lengths:
        print(f"Testing with window length: {window_length}")

        # Extract sliding windows for training data
        X_train, y_train = lib.extract_sliding_windows(train_sequences, train_labels, window_length)

        # Extract sliding windows for testing data
        X_test, y_test = lib.extract_sliding_windows(test_sequences, test_labels, window_length)

        if model == "tcn":
            model = lib.build_tcn_model(window_length)
        else:
            model = lib.build_lstm_model(window_length)

        # Compile the model
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        results.append({"window_length": window_length, "test_loss": test_loss, "test_accuracy": test_accuracy})
        print(f"Window Length: {window_length}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    
    # Display results
    results_df = pd.DataFrame(results)
    results_df.to_csv("results_window_lengths.csv", index=False)
    print("Results saved to results_window_lengths.csv")

def build_tcn_model(hp):
    model = Sequential([
        Input(shape=(best_tcn_window, 1)),  # Define the input shape explicitly
        TCN(
            nb_filters=hp.Int('nb_filters', 16, 128, step=16),
            kernel_size=hp.Choice('kernel_size', [2, 3, 5])
        ),
        Dense(1, activation="sigmoid")  # Binary classification
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_lstm_model(hp):
    model = Sequential([
        Input(shape=(best_lstm_window, 1)),
        LSTM(
            units=hp.Int('units', 16, 128, step=16)
        ),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def tune_hyper(train_data, test_data, window_length, model_func):
    # Parse sequences into lists of integers
    train_sequences = lib.get_seq(train_data['sequence'])
    train_labels = lib.get_labels(train_data['label'])

    test_sequences = lib.get_seq(test_data['sequence'])
    test_labels = lib.get_labels(test_data['label'])

    X_train, y_train = lib.extract_sliding_windows(train_sequences, train_labels, window_length)
    X_test, y_test = lib.extract_sliding_windows(test_sequences, test_labels, window_length)
    tuner = RandomSearch(
        model_func,
        objective="val_accuracy",
        max_trials=100,
        directory="tuning",
        project_name="tcn_adfa_tuning"
    )
    
    tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


if __name__ == "__main__":
    # Load data from CSV files
    train_d = pd.read_csv("data/train_data.csv")
    test_d = pd.read_csv("data/test_data.csv")

    best_window("lstm", train_d, test_d)
    # tune_hyper(train_d, test_d, 5, build_tcn_model)
    # tune_hyper(train_d, test_d, 5, build_lstm_model)

