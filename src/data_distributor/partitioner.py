"""
Helper for shuffling and partitioning initial dataset depending on the number of clients
"""

# run from root: python .\src\data_distributor\partitioner.py
# server validation will only use 100% from appointed data for testing (check use_full_dataset param in centralized evaluation)


# Number of clients: 2
# Row count total: 1194464
# Using server validation split fraction of 0.3 of full dataset
# Using stream split fraction of 0.1 of each clients data
# Using 0.5 of clients data for each backup client
# Generating data for server validation
# Row count server_data: 358339
# Row count clients_data: 836125
# Generating data for backup client 1
# Row count: 418063
# Generating data for backup client 2
# Row count: 418062
# Generating data for client 1
# Row count stream: 41806
# Row count presplit: 376257
# Generating data for client 2
# Row count stream: 41806
# Row count presplit: 376256


import pandas as pd
import numpy as np

filename = "data/full_dataset/wustl_iiot_2021.csv"
num_clients = 2
backup_clients = 2

df = pd.read_csv(filename)
row_count = len(df.index)
random_state = 1
stream_split_fraction = 0.1
server_validation_fraction = 0.3
print("Number of clients: %s" % num_clients)
print("Row count total: %s" % row_count)
print("Using server validation split fraction of %s of full dataset" % server_validation_fraction)
print("Using stream split fraction of %s of each clients data" % stream_split_fraction)
print("Using %s of clients data for each backup client" % str(1/backup_clients))

# shuffle dataset
ds = df.sample(frac=1, random_state = random_state)

print("Generating data for server validation")

server_data, clients_data = np.array_split(ds, [int(server_validation_fraction * len(ds))])

print("Row count server_data: %s" % len(server_data))
print("Row count clients_data: %s" % len(clients_data))
server_data.to_csv(f"data/{num_clients}_client_setup/server/wustl_iiot_2021.csv", index=False)

for i, dfsplit in enumerate(np.array_split(clients_data, backup_clients)):
    print("Generating data for backup client %s" % str(i+1))
    print("Row count: %s" % len(dfsplit))
    dfsplit.to_csv(f"data/{num_clients}_client_setup/backup_{i+1}/wustl_iiot_2021.csv", index=False)
for i, dfsplit in enumerate(np.array_split(clients_data, num_clients)):
    print("Generating data for client %s" % str(i+1))
    stream, presplit = np.array_split(dfsplit, [int(stream_split_fraction * len(dfsplit))])
    print("Row count stream: %s" % len(stream))
    print("Row count presplit: %s" % len(presplit))

    presplit.to_csv(f"data/{num_clients}_client_setup/client_{i+1}/wustl_iiot_2021.csv", index=False)
    stream.to_csv(f"data/{num_clients}_client_setup/streams/data{i+1}_stream.csv", index=False)




