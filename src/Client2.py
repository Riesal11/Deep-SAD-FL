import flwr as fl
import torch
import numpy as np

import logging
from datasets.main import load_dataset
from DeepSAD import DeepSAD

class FL_Client(fl.client.NumPyClient):

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def fit(self, parameters, config):
        deepSAD.train(dataset,device=device,lr=0.0001, n_epochs=150, lr_milestones=(50,), batch_size=128, weight_decay=0.5e-6)
        parameters = self.get_parameters()
        return parameters, num_examples["trainset"]

    def evaluate(self, parameters, config):
        deepSAD.test(dataset, device='cpu')
        auc = deepSAD.results['test_auc']
        loss = deepSAD.results['test_loss']
        print(auc)
        print(loss)
        return float(loss), num_examples["testset"], {"accuracy": float(auc)}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file = '../log/Flower_log/log_client2.txt'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

if not torch.cuda.is_available():
    device = 'cpu'

logger.info('Computation device: %s' % device)


np.random.seed(558)
random_state=np.random.RandomState(558)

dataset = load_dataset(dataset_name = 'iiot', data_path = '../data', normal_class = 0, known_outlier_class = 1, n_known_outlier_classes = 1,
ratio_known_normal = 0.01, ratio_known_outlier = 0.01, ratio_pollution = 0.01, random_state=random_state)
num_examples = {"trainset" : len(dataset.train_set), "testset" : len(dataset.test_set)}
deepSAD = DeepSAD()
deepSAD.set_network(net_name = 'iiot_mlp')
net = deepSAD.net

fl.client.start_numpy_client("localhost:8080", client=FL_Client())
