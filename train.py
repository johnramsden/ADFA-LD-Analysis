import tensorflow as tf
assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()
import pandas as pd

import lib

def train_tcn(train_data, test_data):
    window_length = 95
    nb_filters = 96
    kernel_size = 2

    train_sequences = lib.get_seq(train_data['sequence'])
    train_labels = lib.get_labels(train_data['label'])

    test_sequences = lib.get_seq(test_data['sequence'])
    test_labels = lib.get_labels(test_data['label'])

    # Extract sliding windows for training data
    X_train, y_train = lib.extract_sliding_windows(train_sequences, train_labels, window_length)

    # Extract sliding windows for testing data
    X_test, y_test = lib.extract_sliding_windows(test_sequences, test_labels, window_length)

    # Build the TCN model
    model = lib.build_tcn_model(
        window_length,
        nb_filters=nb_filters,
        kernel_size=kernel_size
    )

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    model.save("./models/tcn_adfa_model.keras")

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

def train_lstm(train_data, test_data):
    window_length = 100
    units = 96

    train_sequences = lib.get_seq(train_data['sequence'])
    train_labels = lib.get_labels(train_data['label'])

    test_sequences = lib.get_seq(test_data['sequence'])
    test_labels = lib.get_labels(test_data['label'])

    # Extract sliding windows for training data
    X_train, y_train = lib.extract_sliding_windows(train_sequences, train_labels, window_length)

    # Extract sliding windows for testing data
    X_test, y_test = lib.extract_sliding_windows(test_sequences, test_labels, window_length)

    # Build the TCN model
    model = lib.build_lstm_model(
        window_length,
        units=units
    )

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    model.save("./models/lstm_adfa_model.keras")

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


if __name__ == '__main__':
    # Load data from CSV files
    train_d = pd.read_csv("data/train_data.csv")
    test_d = pd.read_csv("data/test_data.csv")

    train_tcn(train_d, test_d) # 0.9446, 95w, 64nb, 2k
    train_lstm(train_d, test_d) # 0.9544, 96u