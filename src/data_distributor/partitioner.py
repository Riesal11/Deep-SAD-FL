"""
Helper for shuffling and partitioning initial dataset depending on the number of clients
"""

# Number of clients: 2
# Row count total: 1194464
# Using stream split of 10% of each clients data
# Generating data for client 0
# Row count stream: 59723
# Row count presplit: 537509
# Generating data for client 1
# Row count stream: 59723
# Row count presplit: 537509


import pandas as pd
import numpy as np

filename = "data/full_dataset/wustl_iiot_2021.csv"
num_clients = 2

df = pd.read_csv(filename)
row_count = len(df.index)
random_state = 1
stream_split_fraction = 0.1
print("Number of clients: %s" % num_clients)
print("Row count total: %s" % row_count)
print("Using stream split fraction of %s of each clients data" % stream_split_fraction)
# shuffle dataset
ds = df.sample(frac=1, random_state = random_state)
for i, dfsplit in enumerate(np.array_split(ds, num_clients)):
    print("Generating data for client %s" % str(i+1))
    stream, presplit = np.array_split(dfsplit, [int(stream_split_fraction * len(dfsplit))])
    print("Row count stream: %s" % len(stream))
    print("Row count presplit: %s" % len(presplit))

    presplit.to_csv(f"data/{num_clients}_client_setup/client_{i+1}/wustl_iiot_2021.csv", index=False)
    stream.to_csv(f"data/{num_clients}_client_setup/streams/data{i+1}_stream.csv", index=False)




