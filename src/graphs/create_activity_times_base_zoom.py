
import matplotlib.pyplot as plt
import pandas as pd
import os

color_map = {
    "Startup": "#d3d3d3", 
    "Train": "skyblue",
    "Test": "salmon"
}

colors = {
    "Client 1": "blue",
    "Client 2": "orange",
    "Client 3": "green"
}

data_path = "../../test_results/baseline/base_fed_3_clients_50_epoch/activity_times_base_federated_full_dataset.csv"

plot_dir = "base_activity_times"
csv_dir  = plot_dir+"/metrics_export"
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(csv_dir,  exist_ok=True)

df = pd.read_csv(data_path)


client_order = ["Client 3", "Client 2", "Client 1"]
df["Client"] = pd.Categorical(df["Client"], categories=client_order, ordered=True)
df.sort_values(by="Client", inplace=True)

filtered_df = []

for client in df["Client"].cat.categories:
    client_df = df[df["Client"] == client]
    for phase in ["Startup", "Train", "Test"]:
        first_entry = client_df[client_df["Type"] == phase].sort_values(by="Start").head(1)
        filtered_df.append(first_entry)

df = pd.concat(filtered_df).sort_values(by="Client")

first_train_sec = df[df["Type"] == "Train"]["StartOffsetSec"].min()
finish_times_all = df[df["Type"] == "Train"][["Client", "StartOffsetSec", "DurationSec"]].copy()
finish_times_all["FinishSec"] = finish_times_all["StartOffsetSec"] + finish_times_all["DurationSec"]

fig, ax = plt.subplots(figsize=(12, 6))

for idx, row in df.iterrows():
    color = color_map.get(row["Type"], "lightgray")
    ax.barh(row["Client"], row["DurationSec"], left=row["StartOffsetSec"],
            color=color, edgecolor="black", height=0.8, label=row["Type"])

ax.axvline(first_train_sec, color="red", linestyle="--", linewidth=2)
ymin, ymax = ax.get_ylim()
ax.text(first_train_sec + 2, ymax + 0.5, f"{int(first_train_sec)}s",
        color="red", fontsize=12, ha="left", va="bottom")

for _, row in finish_times_all.iterrows():
    ax.axvline(row["FinishSec"], color=colors[row["Client"]], linestyle="--", linewidth=2)
    ax.text(row["FinishSec"] + 2, ymax + 0.5, f"{int(row['FinishSec'])}s",
            color=colors[row["Client"]], fontsize=12, ha="left", va="bottom")

ax.set_xlabel("Elapsed Time (seconds)", fontsize=18)
ax.set_ylabel("", fontsize=18)
ax.set_title("Activity Times (Base - federated - full dataset)", fontsize=20)
ax.tick_params(axis="both", labelsize=14)
plt.xticks(rotation=45)

legend_order = ["Train", "Test", "Startup"]
legend_colors = [color_map[label] for label in legend_order]
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor='black')
    for c in legend_colors
]
ax.legend(legend_handles, legend_order, fontsize=15)

ax.invert_yaxis()
filename = "activity_times_base_federated_full_dataset_zoom"
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, filename))
plt.close()

print(f"Saved graph to: {os.path.abspath(plot_dir)}")

csv_filename = "activity_times_base_federated_full_dataset_zoom.csv"
df.to_csv(os.path.join(csv_dir, csv_filename), index=False)

print(f"Saved CSVs to: {os.path.abspath(csv_dir)}")
