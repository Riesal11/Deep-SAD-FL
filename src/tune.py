import numpy as np
import ray
from ray import tune
import numpy as np


from datasets.main import load_dataset
from DeepSAD import DeepSAD

def train_tune(config,checkpoint_dir=None):
    
    #dataset = load_dataset('iiot', config['data_path'],config['fl_dataset_index'], config['dataset_size'],config['net_name'],config['normal_class'], config['known_outlier_class'], config['n_known_outlier_classes'],
    #config['ratio_known_outlier'], config['ratio_pollution'], random_state=config['random_state'])
    dataset = load_dataset('iiot', data_path='/content/drive/Othercomputers/My laptop/Deep-SAD-FL/data',
              net_name='iiot_no_cat',normal_class=0, known_outlier_class=1,n_known_outlier_classes=1, 
              ratio_known_outlier=0.01,ratio_pollution=0.05,random_state=np.random.RandomState(1))
    deepSAD = DeepSAD(eta=1)
    deepSAD.set_network('iiot_no_cat',h1=config['h1']) #first hidden layer of the neuronal network

    deepSAD.train(dataset,device='cuda', lr=config['lr'], 
        optimizer_name='adam', n_epochs=50, 
        lr_milestones=(25,), batch_size=config['bs'], 
        weight_decay=5e-07) 
        #optimize for the learning rate and batch size
    
    deepSAD.test(dataset, device='cuda')

    tune.report(f_score=deepSAD.results['test_f1'])
