import os
import re
import pandas as pd
import matplotlib.pyplot as plt

baseline_dir = "../../test_results/baseline"
file_groups = {
    "fed_dos": {
        "client1": baseline_dir + "/base_fed_3_clients_50_epoch/client1/client1_test_results_fed_dos.csv",
        "client2": baseline_dir + "/base_fed_3_clients_50_epoch/client2/client2_test_results_fed_dos.csv",
        "client3": baseline_dir + "/base_fed_3_clients_50_epoch/client3/client3_test_results_fed_dos.csv",
    },
    "local_dos": {
        "client1": baseline_dir + "/base_local_3_clients_50_epoch/client1/client1_test_results_local_dos.csv",
        "client2": baseline_dir + "/base_local_3_clients_50_epoch/client2/client2_test_results_local_dos.csv",
        "client3": baseline_dir + "/base_local_3_clients_50_epoch/client3/client3_test_results_local_dos.csv",
    },
    "fed_all_atk": {
        "client1": baseline_dir + "/base_fed_3_clients_full_atk_50_epoch/client1/client1_test_results_fed_full.csv",
        "client2": baseline_dir + "/base_fed_3_clients_full_atk_50_epoch/client2/client2_test_results_fed_full.csv",
        "client3": baseline_dir + "/base_fed_3_clients_full_atk_50_epoch/client3/client3_test_results_fed_full.csv",
    },
    "local_all_atk": {
        "client1": baseline_dir + "/base_local_3_clients_full_atk1_50_epoch/client1/client1_test_results_local_full.csv",
        "client2": baseline_dir + "/base_local_3_clients_full_atk1_50_epoch/client2/client2_test_results_local_full.csv",
        "client3": baseline_dir + "/base_local_3_clients_full_atk1_50_epoch/client3/client3_test_results_local_full.csv",
    },
}

env_order = ["fed_dos", "local_dos", "fed_all_atk", "local_all_atk"]

env_color_shades = {
    "fed_dos": ["#ff9999", "#ff4d4d", "#b30000"],       # reds
    "local_dos": ["#99ccff", "#3399ff", "#004080"],     # blues
    "fed_all_atk": ["#ffd699", "#ff9933", "#cc6600"],   # oranges
    "local_all_atk": ["#b3ffb3", "#66ff66", "#009933"], # greens
}

plot_dir = "baseline_model_eval_all_envs"
csv_dir  = plot_dir+"/metrics_export"
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(csv_dir,  exist_ok=True)


def extract_column(df: pd.DataFrame, column_name: str):
    if column_name not in df.columns:
        return []
    col = df[column_name]
    if col.dtype == object:
        return col.astype(str).str.rstrip("%").astype(float).tolist()
    return col.astype(float).tolist()

def extract_f1(df: pd.DataFrame):
    if "Best Threshold with the F1-score" not in df.columns:
        return []
    pattern = r'"?[^,]+,\s*([0-9.]+)"?'
    out = []
    for x in df["Best Threshold with the F1-score"]:
        s = str(x)
        m = re.search(pattern, s)
        out.append(float(m.group(1)) if m else None)
    return out


metrics_data = {
    "PR-AUC": {},
    "ROC-AUC": {},
    "F1-score of the normal class": {},
    "F1-score from Best Threshold": {},  
}

for env, clients in file_groups.items():
    for key in metrics_data:
        metrics_data[key].setdefault(env, {})
    for client, path in clients.items():
        df = pd.read_csv(path)
        metrics_data["PR-AUC"][env][client] = extract_column(df, "Test PR-AUC")
        metrics_data["ROC-AUC"][env][client] = extract_column(df, "Test ROC-AUC")
        metrics_data["F1-score of the normal class"][env][client] = extract_column(
            df, "F1-score of the normal class"
        )
        metrics_data["F1-score from Best Threshold"][env][client] = extract_f1(df)

def plot_sorted_legend_clean_ylabel(metric_key: str, outfile_png: str):
    plt.figure(figsize=(10, 6))
    lines, labels = [], []

    title = (
        metric_key.replace("PR-AUC", "AUC-PR")
                  .replace("ROC-AUC", "AUC-ROC")
                  .replace("F1-score from Best Threshold", "F1-Score (Anomalous Class)")
                  .replace("F1-score of the normal class", "F1-Score (Normal Class)")
    )
    ylabel = title
    if title in ("AUC-PR", "AUC-ROC"):
        ylabel += " (%)"
    ylabel = ylabel.replace(" (Normal Class)", "").replace(" (Anomalous Class)", "")

    for env in env_order:
        if env not in metrics_data[metric_key]:
            continue
        env_clients = sorted(metrics_data[metric_key][env].keys())
        for idx, client in enumerate(env_clients):
            values = metrics_data[metric_key][env][client]
            if not values:
                continue
            color_list = env_color_shades.get(env, ["#000000"])
            color = color_list[idx % len(color_list)]
            rounds = list(range(1, len(values) + 1))
            line, = plt.plot(
                rounds, values,
                label=f"{client} ({env})",
                color=color,
                linewidth=2.0 
            )
            lines.append(line)
            labels.append(f"{client} ({env})")

    plt.title(title, fontsize=20)
    plt.xlabel("Round", fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.xticks([1, 2, 3], fontsize=18)
    plt.yticks(fontsize=18)
    if lines:
        plt.legend(lines, labels, fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, outfile_png), dpi=200)
    plt.close()

plot_sorted_legend_clean_ylabel("PR-AUC", "AUC-PR.png")
plot_sorted_legend_clean_ylabel("ROC-AUC", "AUC-ROC.png")
plot_sorted_legend_clean_ylabel("F1-score of the normal class", "F1-Score_Normal_Class.png")
plot_sorted_legend_clean_ylabel("F1-score from Best Threshold", "F1-Score_Anomalous_Class.png")

print(f"Saved plots to: {os.path.abspath(plot_dir)}")

metrics_for_export = {
    "AUC-PR": "PR-AUC",
    "AUC-ROC": "ROC-AUC",
    "F1-Score_Normal_Class": "F1-score of the normal class",
    "F1-Score_Anomalous_Class": "F1-score from Best Threshold",
}

for export_name, key in metrics_for_export.items():
    rows = []
    for env in env_order:
        if env not in metrics_data[key]:
            continue
        for client, values in metrics_data[key][env].items():
            for r, score in enumerate(values, start=1):
                rows.append({
                    "Environment": env,
                    "Client": client,
                    "Round": r,
                    "Score": score
                })
    df_out = pd.DataFrame(rows, columns=["Environment", "Client", "Round", "Score"])
    df_out.to_csv(os.path.join(csv_dir, f"{export_name}.csv"), index=False)

print(f"Saved CSVs to: {os.path.abspath(csv_dir)}")
