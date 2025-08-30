
import pandas as pd
import matplotlib.pyplot as plt
import os

color_map = {
    "startup": "#d3d3d3", 
    "train": "skyblue",
    "test": "salmon"
}

plot_dir = "final_activity_times"
csv_dir  = plot_dir+"/metrics_export"
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(csv_dir,  exist_ok=True)

df = pd.read_csv("../../test_results/final/final_50epoch_3_clients_backup/activity_times_final_federated.csv", parse_dates=["Start", "End"])

client_order = ["Client 1", "Client 2", "Client 3", "Backup 1"]
df["Client"] = pd.Categorical(df["Client"], categories=client_order, ordered=True)
df.sort_values(by=["Client", "Start"], inplace=True)

fig, ax = plt.subplots(figsize=(14, 6))
for _, row in df.iterrows():
    color = color_map.get(row["Type"], "gray")
    ax.barh(row["Client"], row["DurationSec"], left=row["StartOffsetSec"],
            color=color, edgecolor='black', height=0.8, label=row["Type"].capitalize())

first_train_sec = df[df["Type"] == "train"]["StartOffsetSec"].min()
ax.axvline(first_train_sec, color='red', linestyle='--', linewidth=2)
ymin, ymax = ax.get_ylim()
ax.text(first_train_sec + 2, ymax + 0.5, f"{int(first_train_sec)}s",
        color='red', fontsize=15, ha='left', va='bottom')

ax.set_xlim(0, df["EndOffsetSec"].max() + 20)
ax.set_xlabel("Elapsed Time (seconds)", fontsize=18)
ax.set_ylabel("", fontsize=18)
ax.set_title("Activity Times (async - federated)", fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.xticks(rotation=45)
ax.invert_yaxis()

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=15)

filename = "activity_times_final_federated"

plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, filename))
plt.close()

print(f"Saved graph to: {os.path.abspath(plot_dir)}")

csv_filename = "activity_times_final_federated.csv"
df.to_csv(os.path.join(csv_dir, csv_filename), index=False)

print(f"Saved CSVs to: {os.path.abspath(csv_dir)}")
