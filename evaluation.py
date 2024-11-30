import tensorflow as tf

assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM
from tcn import TCN
import lib
from keras_tuner.tuners import RandomSearch
import keras_tuner
from sklearn import model_selection
import numpy as np


def build_lstm_model_cv(hp):
    # Define hyperparameters
    window_length = hp.Int('window_length', 50, 200, step=10)  # Tune window length
    units = hp.Int('units', 16, 128, step=16)             # Number of units

    # Define the model
    model = Sequential([
        Input(shape=(window_length, 1)),  # Input shape uses dynamic window length
        LSTM(units=units),
        Dense(1, activation="sigmoid")  # Binary classification
    ])

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_tcn_model_cv(hp):
    # Define hyperparameters
    window_length = hp.Int('window_length', 50, 200, step=10)  # Tune window length
    nb_filters = hp.Int('nb_filters', 16, 128, step=16)  # Number of filters
    kernel_size = hp.Choice('kernel_size', [2, 3, 5])  # Kernel size

    # Define the model
    model = Sequential([
        Input(shape=(window_length, 1)),  # Input shape uses dynamic window length
        TCN(nb_filters=nb_filters, kernel_size=kernel_size),
        Dense(1, activation="sigmoid")  # Binary classification
    ])

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model



class CVTuner(keras_tuner.engine.tuner.Tuner):
    """
    Custom tuner for cross validation

    Modified from:
    https://freedium.cfd/https://python.plainenglish.io/how-to-do-cross-validation-in-keras-tuner-db4b2dbe079a
    """
    def run_trial(self, trial, x, y, epochs = 10):
        cv = model_selection.KFold(5)
        val_losses = []

        window_length = trial.hyperparameters.get('window_length')
        X, y = lib.extract_sliding_windows(x, y, window_length)

        for train_indices, test_indices in cv.split(x):
            x_train, x_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            model = self.hypermodel.build(trial.hyperparameters)
            model.fit(x_train, y_train, epochs=epochs)
            val_losses.append(model.evaluate(x_test, y_test))

        self.oracle.update_trial(trial.trial_id, {'val_loss': np.mean(val_losses)})

def hyper_optim_cv(train_data, model_func, tag, max_trials=10, epochs=10, oracle=keras_tuner.oracles.BayesianOptimizationOracle):
    train_sequences = lib.get_seq(train_data['sequence'])
    train_labels = lib.get_labels(train_data['label'])

    # Define the tuner
    tuner = CVTuner(
        hypermodel=model_func,
        oracle=oracle(
            objective="val_loss",  # Optimize validation loss
            max_trials=max_trials
        ),
        directory="tuning",
        project_name=f"{tag}_tuning_cv"
    )

    # Perform the search
    tuner.search(
        x=train_sequences,
        y=train_labels,
        epochs=epochs
    )

    # Retrieve the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best hyperparameters:", best_hps.values)

    return best_hps

if __name__ == "__main__":
    # Load data from CSV files
    train_d = pd.read_csv("data/train_data.csv")
    test_d = pd.read_csv("data/test_data.csv")

    best = hyper_optim_cv(train_data=train_d, model_func=build_tcn_model_cv, tag="tcn", max_trials=50)
    print(f"TCN Best {best.values}")

    best = hyper_optim_cv(train_data=train_d, model_func=build_lstm_model_cv, tag="lstm", max_trials=50)
    print(f"LSTM Best {best.values}")




