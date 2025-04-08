import csv
import os

def write_results_to_csv(
    file_path, test_loss, anomaly_scores_min, anomaly_scores_max,
    best_threshold_f1, test_precision, test_recall, test_precision_norm,
    test_recall_norm, test_f1_norm, test_pr_auc, test_roc_auc, test_time
):
    headers = [
        "Test Loss",
        "Anomaly scores min",
        "Anomaly scores max",
        "Best Threshold with the F1-score",
        "Precision",
        "Recall",
        "Precision of the normal class",
        "Recall of the normal class",
        "F1-score of the normal class",
        "Test PR-AUC",
        "Test ROC-AUC",
        "Test Time"
    ]

    row = [
        f"{test_loss:.6f}",
        f"{anomaly_scores_min:.6f}",
        f"{anomaly_scores_max:.6f}",
        f'"{best_threshold_f1}"',
        f"{test_precision:.6f}",
        f"{test_recall:.6f}",
        f"{test_precision_norm:.6f}",
        f"{test_recall_norm:.6f}",
        f"{test_f1_norm:.6f}",
        f"{test_pr_auc:.2f}%",
        f"{test_roc_auc:.2f}%",
        f"{test_time:.3f}"
    ]

    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

        if not file_exists:
            writer.writerow(headers)

        writer.writerow(row)

    print(f"Results successfully written to {file_path}")