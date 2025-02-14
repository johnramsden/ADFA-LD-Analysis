from keras.src.layers import LSTM
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tcn import TCN
import tensorflow as tf
import random, os

# Set the seed for TensorFlow
tf.random.set_seed(42)
# Set the seed for NumPy
np.random.seed(42)
# Set the seed for Python's random module
random.seed(42)
# Optionally set the seed for the OS
os.environ['PYTHONHASHSEED'] = str(42)

def parse_sequence(sequence):
    """
    Parse a sequence string into a list of integers
    """
    # Remove any surrounding brackets and split on whitespace
    cleaned_sequence = sequence.strip('[]').replace(',', ' ')  # Handle both commas and spaces
    try:
        return list(map(int, cleaned_sequence.split()))  # Split by spaces and convert to integers
    except ValueError:
        raise ValueError(f"Invalid sequence format: {sequence}")


def extract_sliding_windows(data: list[list[int]], labels: list, window_length: int) -> tuple:
    """
    Extract sliding windows from sequences with labels.

    :param data: List of sequences, where each sequence is a list of integers.
    :type data: list[list[int]]
    :param labels: Corresponding labels for the sequences.
    :type labels: list
    :param window_length: Fixed length for each sliding window.
    :type window_length: int
    :return: A tuple containing:
             - X: Array of sliding windows with shape (num_samples, window_length, 1).
             - y: Corresponding labels for each window.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    X = []
    y = []

    for sequence, label in zip(data, labels):
        # These overlap
        for i in range(len(sequence) - window_length + 1):
            window = sequence[i:i + window_length]
            X.append(window)
            y.append(label)

    # Convert to numpy arrays and expand dimensions for TCN/LSTM input
    X = np.array(X)
    X = np.expand_dims(X, axis=-1)  # Add feature dimension
    y = np.array(y)

    return X, y


def build_tcn_model(window_length, nb_filters=32, kernel_size=2):
    model = Sequential([
        Input(shape=(window_length, 1)),
        TCN(
            nb_filters=nb_filters,
            kernel_size=kernel_size
        ),
        Dense(1, activation="sigmoid")  # Binary classification
    ])
    return model

def build_lstm_model(window_length, units=96):
    model = Sequential([
        Input(shape=(window_length, 1)),
        LSTM(
            units=units
        ),
        Dense(1, activation="sigmoid")
    ])
    return model

def get_seq(df):
    """
    Parse a sequence cleanly into individual integer list
    """
    return df.apply(parse_sequence).tolist()

def get_labels(df):
    """
    Get integer labels
    """
    return df.apply(lambda x: 0 if x == "normal" else 1).tolist()
