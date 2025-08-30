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

To build the image, in the application root folder, execute
```
docker build -t riesal11/deep-sad-fl:base .
```
To run the the container
```
docker run -t -d -p 8080:8080 riesal11/deep-sad-fl:base
docker run -t -d -p 3000:3000 riesal11/deep-sad-fl:base
```
To push the image to the docker hub
```
docker push riesal11/deep-sad-fl:base
```

From the client machines, pull the image
```
docker pull riesal11/deep-sad-fl:base
```

Run the containers
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


## Docker

