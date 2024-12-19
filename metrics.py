import os

import lib
import pandas as pd
import keras
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    f1_score, recall_score, accuracy_score, precision_score,
    classification_report)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import time


def test_tcn(xtst, ytst):
    model = keras.saving.load_model("models/tcn_adfa_model.keras")

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(xtst, ytst, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return model

def test_lstm(xtst, ytest):
    model = keras.saving.load_model("models/lstm_adfa_model.keras")

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(xtst, ytest, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return model

import seaborn as sns
import numpy as np

def plot_confusion_matrix_with_details(cm, labels, title, save_path):
    # Calculate percentages
    cm_percentage = cm / cm.sum(axis=1, keepdims=True) * 100

    # Create combined matrix with actual counts, percentages, and FN/FP/TP labels
    combined_matrix = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == 0 and j == 0:  # (TN)
                cell_label = f"TN: {cm[i, j]}\n({cm_percentage[i, j]:.2f}%)"
            elif i == 1 and j == 1:  # TP
                cell_label = f"TP: {cm[i, j]}\n({cm_percentage[i, j]:.2f}%)"
            else:  # Off-diagonal
                if i == 1 and j == 0:  # False Positives (FP)
                    cell_label = f"FN: {cm[i, j]}\n({cm_percentage[i, j]:.2f}%)"
                else:  # False Negatives (FN)
                    cell_label = f"FP: {cm[i, j]}\n({cm_percentage[i, j]:.2f}%)"
            combined_matrix[i, j] = cell_label

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=combined_matrix, fmt='', cmap='Blues',
        xticklabels=labels, yticklabels=labels, cbar=False
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__":
    test_data = pd.read_csv("data/test_data.csv")

    window_length_tcn = 200
    window_length_lstm = 190

    test_sequences = lib.get_seq(test_data['sequence'])
    test_labels = lib.get_labels(test_data['label'])

    # Extract sliding windows for testing data
    x_test_tcn, y_test_tcn = lib.extract_sliding_windows(test_sequences, test_labels, window_length_tcn)
    x_test_lstm, y_test_lstm = lib.extract_sliding_windows(test_sequences, test_labels, window_length_lstm)

    tcn_model = test_tcn(x_test_tcn, y_test_tcn)
    tcn_predictions = tcn_model.predict(x_test_tcn)

    lstm_model = test_lstm(x_test_lstm, y_test_lstm)
    lstm_predictions = lstm_model.predict(x_test_lstm)

    # Time

    start_tcn = time.time()
    for _ in range(0, 6):
        tcn_predictions = tcn_model.predict(x_test_tcn)
    end_tcn = time.time()

    start_lstm = time.time()
    for _ in range(0, 6):
        lstm_predictions = lstm_model.predict(x_test_lstm)
    end_lstm = time.time()

    tcn_time = (end_tcn - start_tcn) / 5
    lstm_time = (end_lstm - start_lstm) / 5

    plt.bar(["TCN", "LSTM"], [tcn_time, lstm_time])
    plt.ylabel("Time (seconds)")
    plt.title("Prediction Time Comparison")
    # plt.show()
    plt.savefig(f"figures/prediction_time.png")
    plt.close('all')

    # Convert probabilities to class labels
    for threshold in [0.5, 0.9, 0.95]:
        if not os.path.exists(f"figures/{threshold}"):
            os.mkdir(f"figures/{threshold}")

        print(f"# --- Threshold {threshold} --- #")

        tcn_class_predictions = (tcn_predictions > threshold).astype(int)
        lstm_class_predictions = (lstm_predictions > threshold).astype(int)

        # Plot CM

        cm_tcn = confusion_matrix(y_test_tcn, tcn_class_predictions)
        unique_labels = sorted(set(y_test_tcn.flatten().tolist()) | set(tcn_class_predictions.flatten().tolist()))
        print("Original Labels TCN:", unique_labels)
        cm_lstm = confusion_matrix(y_test_lstm, lstm_class_predictions)
        unique_labels = sorted(set(y_test_lstm.flatten().tolist()) | set(lstm_class_predictions.flatten().tolist()))
        print("Original Labels LSTM:", unique_labels)

        plot_confusion_matrix_with_details(
            cm_tcn, ["Normal", "Abnormal"], "Confusion Matrix (TCN)",
            f"figures/{threshold}/tcn_confusion_matrix_both.png"
        )

        plt.close('all')

        plot_confusion_matrix_with_details(
            cm_lstm, ["Normal", "Abnormal"], "Confusion Matrix (LSTM)",
            f"figures/{threshold}/lstm_confusion_matrix_both.png"
        )

        plt.close('all')

        # plot ROC

        fpr_tcn, tpr_tcn, _ = roc_curve(y_test_tcn, tcn_predictions)
        fpr_lstm, tpr_lstm, _ = roc_curve(y_test_lstm, lstm_predictions)

        auc_tcn = auc(fpr_tcn, tpr_tcn)
        auc_lstm = auc(fpr_lstm, tpr_lstm)

        plt.plot(fpr_tcn, tpr_tcn, label=f"TCN (AUC={auc_tcn:.2f})")
        plt.plot(fpr_lstm, tpr_lstm, label=f"LSTM (AUC={auc_lstm:.2f})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(f"figures/{threshold}/roc_curve.png")
        # plt.show()
        plt.close('all')

        # Precision-Recall Curve

        precision_tcn, recall_tcn, _ = precision_recall_curve(y_test_tcn, tcn_predictions)
        precision_lstm, recall_lstm, _ = precision_recall_curve(y_test_lstm, lstm_predictions)

        plt.plot(recall_tcn, precision_tcn, label="TCN")
        plt.plot(recall_lstm, precision_lstm, label="LSTM")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.savefig(f"figures/{threshold}/precision_recall_curve.png")
        # plt.show()
        plt.close('all')

        # RAW F1 and Recall

        tcn_f1 = f1_score(y_test_tcn, tcn_class_predictions)
        tcn_recall = recall_score(y_test_tcn, tcn_class_predictions)
        tcn_accuracy = accuracy_score(y_test_tcn, tcn_class_predictions)
        tcn_precision = precision_score(y_test_tcn, tcn_class_predictions)

        print(f"TCN Overall Accuracy: {tcn_accuracy:.4f}")
        print(f"TCN F1 Score: {tcn_f1:.4f}")
        print(f"TCN Recall: {tcn_recall:.4f}")
        print(f"TCN Precision: {tcn_precision:.4f}")
        print(f"TCN Inference Time (s): {tcn_time:.4f}")

        lstm_f1 = f1_score(y_test_lstm, lstm_class_predictions)
        lstm_recall = recall_score(y_test_lstm, lstm_class_predictions)
        lstm_accuracy = accuracy_score(y_test_lstm, lstm_class_predictions)
        lstm_precision = precision_score(y_test_lstm, lstm_class_predictions)

        print(f"LSTM Overall Accuracy: {lstm_accuracy:.4f}")
        print(f"LSTM F1 Score: {lstm_f1:.4f}")
        print(f"LSTM Recall: {lstm_recall:.4f}")
        print(f"LSTM Precision: {lstm_precision:.4f}")
        print(f"LSTM Inference Time (s): {lstm_time:.4f}")

        # Classification reports
        tcn_report_dict = classification_report(y_test_tcn, tcn_class_predictions, output_dict=True)
        lstm_report_dict = classification_report(y_test_lstm, lstm_class_predictions, output_dict=True)

        tcn_report_df = pd.DataFrame(tcn_report_dict).transpose()
        lstm_report_df = pd.DataFrame(lstm_report_dict).transpose()

        print("TCN Classification Report:")
        print(tcn_report_df)

        print("\nLSTM Classification Report:")
        print(lstm_report_df)
