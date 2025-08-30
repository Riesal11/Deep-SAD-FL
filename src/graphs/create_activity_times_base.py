
import matplotlib.pyplot as plt
import pandas as pd
import os

color_map = {
    "Train": "skyblue",
    "Test": "salmon",
    "Startup": "#d3d3d3"
}

plot_dir = "base_activity_times"
csv_dir  = plot_dir+"/metrics_export"
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(csv_dir,  exist_ok=True)

df = pd.read_csv("../../test_results/baseline/base_fed_3_clients_50_epoch/activity_times_base_federated_full_dataset.csv")

client_order = ["Client 3", "Client 2", "Client 1"]
df["Client"] = pd.Categorical(df["Client"], categories=client_order, ordered=True)
df.sort_values(by="Client", inplace=True)

fig, ax = plt.subplots(figsize=(12, 6))

for idx, row in df.iterrows():
    color = color_map.get(row["Type"], "lightgray")
    ax.barh(
        row["Client"],
        row["DurationSec"],
        left=row["StartOffsetSec"],
        color=color,
        edgecolor='black',
        linewidth=0.5,
        height=0.8,
        label=row["Type"]
    )

ax.set_xlabel("Elapsed Time (seconds)", fontsize=18)
ax.set_ylabel("", fontsize=18)
ax.set_title("Activity Times (Base - federated)", fontsize=20)
ax.tick_params(axis='both', labelsize=18)
plt.xticks(rotation=45)
ax.invert_yaxis()

legend_order = ["Train", "Test", "Startup"]
legend_colors = [color_map[label] for label in legend_order]
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor='black')
    for c in legend_colors
]
ax.legend(legend_handles, legend_order, fontsize=15)

filename = "activity_times_base_federated_full_dataset"

plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, filename))
plt.close()

print(f"Saved graph to: {os.path.abspath(plot_dir)}")

csv_filename = "activity_times_base_federated_full_dataset.csv"
df.to_csv(os.path.join(csv_dir, csv_filename), index=False)

print(f"Saved CSVs to: {os.path.abspath(csv_dir)}")