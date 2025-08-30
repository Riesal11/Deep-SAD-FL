
import pandas as pd
import matplotlib.pyplot as plt
import os

file_dir = "../../test_results/final/final_50epoch_3_clients_backup/timestamped_result_files/"
plot_dir = "final_model_eval"
csv_dir  = plot_dir+"/metrics_export"
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(csv_dir,  exist_ok=True)

fixed_start_time = pd.to_datetime("2025-04-07 21:07:34.747", format="%Y-%m-%d %H:%M:%S.%f")
custom_colors = {
    "Server": "blue",
    "Client 1": "orange",
    "Client 2": "green",
    "Client 3": "red",
    "Backup 1": "purple"
}

def recalculate_elapsed_time(df):
    timestamps = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S,%f")
    df["Elapsed Time (s)"] = (timestamps - fixed_start_time).dt.total_seconds()
    return df

def extract_metric(df, label, column, value_extractor=None):
    if value_extractor:
        metric = df[column].apply(value_extractor)
    else:
        metric = df[column].astype(float)
    return pd.DataFrame({
        "Elapsed Time (s)": df["Elapsed Time (s)"],
        "Metric": metric,
        "Label": label
    })

def plot_and_save(df, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    for label, color in custom_colors.items():
        subset = df[df["Label"] == label]
        plt.plot(subset["Elapsed Time (s)"], subset["Metric"], marker='o', label=label, color=color)
    plt.title(title, fontsize=20)
    plt.xlabel("Elapsed Time (seconds)", fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

files = {
    "Server": file_dir + "server_results_timestamped.csv",
    "Client 1": file_dir + "client1_results_timestamped.csv",
    "Client 2": file_dir + "client2_results_timestamped.csv",
    "Client 3": file_dir + "client3_results_timestamped.csv",
    "Backup 1": file_dir + "backup1_results_timestamped.csv"
}
dfs = {label: recalculate_elapsed_time(pd.read_csv(fname)) for label, fname in files.items()}

pr_auc = pd.concat([extract_metric(dfs[k], k, "Test PR-AUC", lambda x: float(str(x).replace('%', ''))) for k in dfs])
f1_normal = pd.concat([extract_metric(dfs[k], k, "F1-score of the normal class") for k in dfs])
roc_auc = pd.concat([extract_metric(dfs[k], k, "Test ROC-AUC", lambda x: float(str(x).replace('%', ''))) for k in dfs])
f1_anomalous = pd.concat([extract_metric(dfs[k], k, "Best Threshold with the F1-score", lambda x: float(str(x).split(",")[1].replace('"', '').strip())) for k in dfs])

plot_and_save(pr_auc, "AUC-PR", "AUC-PR (%)", "AUC_PR.png")
plot_and_save(f1_normal, "F1-Score (Normal Class)", "F1-Score", "F1-Score_Normal_Class.png")
plot_and_save(roc_auc, "AUC-ROC", "AUC-ROC (%)", "AUC_ROC.png")
plot_and_save(f1_anomalous, "F1-Score (Anomalous Class)", "F1-Score", "F1-Score_Anomalous_Class.png")

print(f"Saved plots to: {os.path.abspath(plot_dir)}")

pr_auc.to_csv(csv_dir+"/AUC-PR.csv", index=False)
f1_normal.to_csv(csv_dir+"/F1-Score_Normal_Class.csv", index=False)
roc_auc.to_csv(csv_dir+"/AUC_ROC.csv", index=False)
f1_anomalous.to_csv(csv_dir+"/F1-Score_Anomalous_Class.csv", index=False)

print(f"Saved CSVs to: {os.path.abspath(csv_dir)}")