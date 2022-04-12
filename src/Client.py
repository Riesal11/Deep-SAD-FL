import flwr as fl


class FL_Client(fl.client.NumPyClient):

    def __init__(self,model,dataset,net,config,device,n_jobs_dataloader):
        self.model = model
        self.dataset = dataset
        self.net = net
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

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def fit(self, parameters, config):

        self.model.train(self.dataset,
                        optimizer_name=self.optimizer_name,
                        device=self.device,
                        lr=self.lr,
                        n_epochs=self.n_epochs, 
                        lr_milestones=self.lr_milestones,
                        batch_size=self.batch_size,
                        weight_decay=self.weight_decay,
                        n_jobs_dataloader=self.n_jobs_dataloader)
        parameters = self.get_parameters()
        return parameters, self.num_examples["trainset"]

    def evaluate(self, parameters, config):
        self.model.test(self.dataset,device=self.device, n_jobs_dataloader=self.n_jobs_dataloader)
        loss = self.model.results['test_loss']
        return float(loss), self.num_examples["testset"], {}
