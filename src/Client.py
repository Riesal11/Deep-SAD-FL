import flwr as fl
import numpy as np
import torch
from collections import OrderedDict
from typing import Dict, List, Tuple
from time import time


class FL_Client(fl.client.NumPyClient):

    def __init__(self,model,dataset,config,device,n_jobs_dataloader):
        self.model = model
        self.dataset = dataset
        self.net = self.model.net
        self.optimizer_name = config['optimizer_name']
        self.lr = config['lr']
        self.device = device
        self.n_epochs = config['n_epochs']
        self.lr_milestones = config['lr_milestone']
        self.batch_size = config['batch_size']
        self.weight_decay = config['weight_decay']
        self.num_examples = {"trainset" : len(dataset.train_set), 
                            "testset" : len(dataset.test_set)}
        self.n_jobs_dataloader = n_jobs_dataloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict)
        #print("Model's state_dict:")
        #for param_tensor in self.model.net.state_dict():
        #    print(param_tensor, "\t", self.model.net.state_dict()[param_tensor].size())


    def fit(self, parameters, config):
        self.set_parameters(parameters)
        start = time()
        self.model.train(self.dataset,
                        optimizer_name=self.optimizer_name,
                        device=self.device,
                        lr=self.lr,
                        n_epochs=self.n_epochs, 
                        lr_milestones=self.lr_milestones,
                        batch_size=self.batch_size,
                        weight_decay=self.weight_decay,
                        n_jobs_dataloader=self.n_jobs_dataloader)
        end = time()
        local_train_time = end - start
        parameters = self.get_parameters(config)
        self.model.test(self.dataset,device=self.device, n_jobs_dataloader=self.n_jobs_dataloader)
        return parameters, self.num_examples["trainset"], {
            "t_diff": local_train_time
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.test(self.dataset,device=self.device, n_jobs_dataloader=self.n_jobs_dataloader)
        loss = self.model.results['test_loss']
        return float(loss), self.num_examples["testset"], {}
        # TODO: for history?
        # return float(loss), self.num_examples["testset"], {self.model.results}
