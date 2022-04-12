import numpy as np
import ray
from ray import tune


from datasets.main import load_dataset
from DeepSAD import DeepSAD

def train_tune(config,checkpoint_dir=None):
    
    dataset = load_dataset('iiot', config['data_path'],config['fl_dataset_index'], config['dataset_size'],config['net_name'],config['normal_class'], config['known_outlier_class'], config['n_known_outlier_classes'],
    config['ratio_known_outlier'], config['ratio_pollution'], random_state=config['random_state'])
    deepSAD = DeepSAD(config['eta'])
    deepSAD.set_network(config['net_name'],h1=config['h1']) #first hidden layer of the neuronal network

    deepSAD.train(dataset,device=config['device'], lr=config['lr'], 
        optimizer_name=config['optimizer_name'], n_epochs=config['n_epochs'], 
        lr_milestones=config['lr_milestone'], batch_size=config['bs'], 
        weight_decay=config['weight_decay'],n_jobs_dataloader=config['n_jobs_dataloader']) 
        #optimize for the learning rate and batch size
    
    deepSAD.test(dataset, device=config['device'], n_jobs_dataloader=config['n_jobs_dataloader'])

    tune.report(f_score=deepSAD.results['test_f1'])
