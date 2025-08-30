# Deep SAD: A Method for Deep Semi-Supervised Anomaly Detection
Original framework: https://github.com/lukasruff/Deep-SAD-PyTorch
Deep SAD FL: Federated Learning enabled 
Adapted framework by veronikaBek: https://github.com/veronikaBek/Deep-SAD-FL

The original framwork was changed and received the following additional capabilities:
1) Docker setup and docker-compose files (`root`).
2) Asynchronous model aggregation (`src/flower_async/`)
3) Backup mechanisms using `Kafka` (`src/data_distributor` and `main.py`)
4) Dynamic data utilization (datasets can change between rounds due to data streams) (`src/data_distributor` and `main.py`)
5) centralized model evaluation on server (`src/flower_async/async_server`)
6) Dataset partitioner helper (`src/data_distributor/partitioner.py`)
7) Graph generation scripts (`src/graphs/`)

## Initial Documentation by veronikaBek
For local setup and documentation, check the documentation by veronikaBek. Note: There is no guarantee that everything still applies/works.

## Local Installation
This code is written in `Python 3.7` and has been adapted to run in `Python 3.12` and requires the packages listed in `requirements.txt`.

Clone the repository to your machine and directory of choice:
```
https://github.com/Riesal11/Deep-SAD-FL.git
```

To run the code locally, we recommend using `conda`:

#### `conda`
```
cd <path-to-Deep-SAD-PyTorch-directory>
conda create --name myenv
source activate myenv
while read requirement; do conda install -n myenv --yes $requirement; done < requirements.txt
```

### Running experiments

First start the server instance in one terminal
```
python main.py --fl_mode server --server_ip_address [::]:8080
```

Then, start a new terminal for each client (min 2 clients)
```
python main.py --fl_mode client --seed 1 --server_ip_address localhost:8080;
python main.py --fl_mode client --seed 2 --server_ip_address localhost:8080
```

Have a look into `main.py` for all possible run arguments and options. As this has not been run locally for quite some time (as the new docker setup is used), it is not guaranteed to work out of the box.

## Branches

For each type of experiment, the is a branch called experiment/{setup}. The master branch is the final setup including async behaviour and backup clients.
If you want to use the experiment with all attack types, look for ...all_types branches.
The working baseline setup, as used by veronikaBek, can be found in experiment/baseline.

## Docker

NOTE: Because the image includes the code to run the application, you first need to adapt the server ip address. Look for all occurences of `<server_ip>` and change it accordingly.

To build the client image, in the application root folder, execute
```
docker buildx build --platform=linux/amd64,linux/arm64 -t {DOCKER_USER}/deep-sad-fl:final -f .\Dockerfile.client .
docker buildx build --platform=linux/amd64,linux/arm64 -t {DOCKER_USER}/deep-sad-fl-backup -f .\Dockerfile.backup .

# image tags for deep-sad-fl: base, final. NOTE: if you want to build the base tag, switch to the experiment/baseline branch first
# image tags for deep-sad-fl-backup: final

```
To run the the container
```
docker run -t -d -p 3000:3000 {DOCKER_USER}/deep-sad-fl:{TAG}
docker run -t -d -p 3003:3003 {DOCKER_USER}/deep-sad-fl-backup:{TAG}


on RPi:
sudo docker run -e SEED=3 -e PORT=3000 --mount type=bind,src=./data/3_client_setup/client_3,dst=/app/data --mount type=bind,src=./log/3_client_setup/client_3,dst=/app/log -t -d -p 3000:3000 {DOCKER_USER}/deep-sad-fl:{TAG}
```

This will automatically start the main.py with respective arguments (as defined by the docker-entrypoint files)


To push the image to the docker hub
```
docker push {DOCKER_USER}/deep-sad-fl:{TAG}
docker push {DOCKER_USER}/deep-sad-fl-backup:{TAG}
```

From the client machines, pull the image
```
docker pull {DOCKER_USER}/deep-sad-fl:{TAG}
docker pull {DOCKER_USER}/deep-sad-fl-backup:{TAG}
```

### docker-compose

Docker needs some environment variables due to the different initial datasets, so place `CLIENT_ID`, `BACKUP_CLIENT_ID`, `CLIENTS` inside root `.env` file, e.g
```
CLIENT_ID=1
BACKUP_CLIENT_ID=1
CLIENTS=3
```

Using server setup (server + 1 backup client + data distributor)
```
docker-compose up -d --build
```
If additional client needed (server + 1 client + 1 backup client + data distributor)
```
docker-compose -f .\docker-compose-server-client.yaml up -d
```
Starting only a client
```
docker-compose -f .\docker-compose-client.yaml up -d
```
Starting only the backup client
```
docker-compose -f .\docker-compose-backup-client.yaml up -d
```
Starting only the server
```
docker-compose -f .\docker-compose-server.yaml up -d
```
Starting only the distributor
```
docker-compose -f .\docker-compose-server.yaml up -d
```
Using a local setup with 2 clients
```
docker-compose -f .\docker-compose-full-local-2.yaml up -d
```
Using a local setup with 3 clients
```
docker-compose -f .\docker-compose-full-local-3.yaml up -d
```

## Experiment result files

After running an experiment, the server and clients have their respective logs and model results stored in `log/{CLIENTS}_client_setup/`. Use them to copy it to the `test_results` folder for future usage in graphs.

## Graphs

Additional graphs created from our experiments are saved in `src/graphs/`. The scripts to create the graphs can also be found in there. They always use data from our experiments, which are saved in `test-results/` (you have to manually add them and change graph scripts to use the correct path).

## Example Run

This setup uses 3 devices (PC, Laptop, Raspberry Pi). The pc runs server, client 1, backup 1 and data distributor. The laptop runs client 2 and backup 2. The Rpi runs client 3.

1) download dataset to `data/full_dataset`

2) Create data and log folders

e.g `log/3_client_setup` folder with empty `backup_1`, `backup_2`, `client_1`, `client_2`, `client_3`, `server` folders. same for `data/3_client_setup`.

3) Create datasets

```
python .\src\data_distributor\partitioner.py
```

4) Create client image 

```
docker buildx build --platform=linux/amd64,linux/arm64 -t {DOCKER_USER}/deep-sad-fl-client:{TAG} -f .\Dockerfile.client .
```
5) Create backup client image 

```
docker buildx build --platform=linux/amd64,linux/arm64 -t {DOCKER_USER}/deep-sad-fl-backup:{TAG} -f .\Dockerfile.backup .
```

6) Copy the respective partitioned datasets to the client 

7) Pull image

client 2 (laptop): 
```
docker pull {DOCKER_USER}/deep-sad-fl-backup
docker pull {DOCKER_USER}/deep-sad-fl:final
```
client 3 (rpi):
```
sudo docker pull {DOCKER_USER}/deep-sad-fl:final
```
8) Start server + client 1 compose 

```
docker-compose -f .\docker-compose-server-client.yaml up -d --build
```

9) Start clients depending on if it is the final setup or base setup

FOR FINAL: Start Clients

client 2 (laptop): 
```
docker-compose -f .\docker-compose-client.yaml up -d
docker-compose -f .\docker-compose-backup-client.yaml up -d
```
client 3 (rpi):
```
sudo docker run -e SEED=3 -e PORT=3000 --mount type=bind,src=./data/3_client_setup/client_3,dst=/app/data --mount type=bind,src=./log/3_client_setup/client_3,dst=/app/log -t -d -p 3000:3000 {DOCKER_USER}/deep-sad-fl:{TAG}
```
OR

FOR BASE: Start Clients
```
docker run -t -d -p 8080:8080 {DOCKER_USER}/deep-sad-fl:base (server)
docker run -t -d -p 3000:3000 {DOCKER_USER}/deep-sad-fl:base	(client 1)
docker run -t -d -p 3001:3001 {DOCKER_USER}/deep-sad-fl:base	(client 2)
docker run -t -d -p 3002:3002 {DOCKER_USER}/deep-sad-fl:base	(client 3)
```

Since in the base branch, you have to manually start the program, in the respective containers, run:
```
python main.py --fl_mode server --server_ip_address [::]:8080
python main.py --fl_mode client --seed 1 --server_ip_address {SERVER_IP}:8080
python main.py --fl_mode client --seed 2 --server_ip_address {SERVER_IP}:8080
python main.py --fl_mode client --seed 3 --server_ip_address {SERVER_IP}:8080
```
## License
MIT
