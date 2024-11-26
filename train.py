import tensorflow as tf
assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()
import pandas as pd

import lib

def train_tcn(train_data, test_data):
    window_length = 20
    nb_filters = 32
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

    model.save("./tcn_adfa_model.keras")

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

def train_lstm(train_data, test_data):
    window_length = 90
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

    model.save("./lstm_adfa_model.keras")

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

from sklearn.model_selection import KFold
import numpy as np

def train_tcn_with_cv(train_data, test_data, k=5):
    window_length = 20
    nb_filters = 32
    kernel_size = 2

    # Prepare sequences and labels
    train_sequences = lib.get_seq(train_data['sequence'])
    train_labels = lib.get_labels(train_data['label'])

    test_sequences = lib.get_seq(test_data['sequence'])
    test_labels = lib.get_labels(test_data['label'])

    # Initialize cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_sequences)):
        print(f"\n--- Fold {fold + 1}/{k} ---")

        # Split data into fold_train and fold_val
        fold_train_sequences = [train_sequences[i] for i in train_idx]
        fold_train_labels = [train_labels[i] for i in train_idx]
        fold_val_sequences = [train_sequences[i] for i in val_idx]
        fold_val_labels = [train_labels[i] for i in val_idx]

        # Extract sliding windows for train and validation
        X_train, y_train = lib.extract_sliding_windows(fold_train_sequences, fold_train_labels, window_length)
        X_val, y_val = lib.extract_sliding_windows(fold_val_sequences, fold_val_labels, window_length)

        # Build the TCN model
        model = lib.build_tcn_model(
            window_length,
            nb_filters=nb_filters,
            kernel_size=kernel_size
        )

        # Compile the model
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=1)

        # Evaluate the model on validation data
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        fold_accuracies.append(val_accuracy)
        print(f"Fold {fold + 1} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Report average performance across folds
    avg_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    print(f"\nAverage Cross-Validation Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")

    # Final evaluation on test data
    print("\n--- Final Evaluation on Test Data ---")
    X_test, y_test = lib.extract_sliding_windows(test_sequences, test_labels, window_length)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    model.save("./tcn_adfa_model.keras")

def train_lstm_with_cv(train_data, test_data, k=5):
    window_length = 90
    units = 96

    # Prepare sequences and labels
    train_sequences = lib.get_seq(train_data['sequence'])
    train_labels = lib.get_labels(train_data['label'])

    test_sequences = lib.get_seq(test_data['sequence'])
    test_labels = lib.get_labels(test_data['label'])

    # Initialize cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_sequences)):
        print(f"\n--- Fold {fold + 1}/{k} ---")

        # Split data into fold_train and fold_val
        fold_train_sequences = [train_sequences[i] for i in train_idx]
        fold_train_labels = [train_labels[i] for i in train_idx]
        fold_val_sequences = [train_sequences[i] for i in val_idx]
        fold_val_labels = [train_labels[i] for i in val_idx]

        # Extract sliding windows for train and validation
        X_train, y_train = lib.extract_sliding_windows(fold_train_sequences, fold_train_labels, window_length)
        X_val, y_val = lib.extract_sliding_windows(fold_val_sequences, fold_val_labels, window_length)

        # Build the LSTM model
        model = lib.build_lstm_model(
            window_length,
            units=units
        )

        # Compile the model
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=1)

        # Evaluate the model on validation data
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        fold_accuracies.append(val_accuracy)
        print(f"Fold {fold + 1} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Report average performance across folds
    avg_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    print(f"\nAverage Cross-Validation Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")

    # Final evaluation on test data
    print("\n--- Final Evaluation on Test Data ---")
    X_test, y_test = lib.extract_sliding_windows(test_sequences, test_labels, window_length)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    model.save("./tcn_adfa_model.keras")



if __name__ == '__main__':
    # Load data from CSV files
    train_d = pd.read_csv("data/train_data.csv")
    test_d = pd.read_csv("data/test_data.csv")

    # train_tcn(train_d, test_d) # 0.8768
    # train_lstm(train_d, test_d) # 0.9513

    train_lstm_with_cv(train_d, test_d)
    train_tcn_with_cv(train_d, test_d)