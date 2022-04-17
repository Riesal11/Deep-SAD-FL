from email.policy import default
import click
import torch
import logging
import random
import numpy as np

import flwr as fl
from Client import FL_Client


import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from tune import train_tune

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from DeepSAD import DeepSAD
from datasets.main import load_dataset


################################################################################
# Settings
################################################################################
@click.command()
@click.option('--hp_tune', type=bool, default=False,
              help='hyperparameter tuning for the iiot dataset.No model training, exits after finding the best parameters')
@click.option('--fl_mode',  type=click.Choice(['off', 'client', 'server']), default='off',
              help='Run Deep-SAD in centralized mode, federated client or federated server')
@click.option('--fl_num_rounds', type=int, default=3,
              help='Number of rounds for federated learning (only when fl_mode = server)')
@click.option('--fl_dataset_index', type=int, default=-1,
              help='''the date is sampled starting from the defined index up to (index+dataset_size)
              If -1 random sample from the dataset with the random seed''')
@click.option('--dataset_name', type=click.Choice(['mnist', 'fmnist', 'cifar10', 'arrhythmia', 'cardio', 'satellite',
                                                   'satimage-2', 'shuttle', 'thyroid', 'iiot']),default='iiot')
@click.option('--dataset_size', type=int, default=-1,
              help='Defines the size of dataset (only for the iiot dataset). Default -1 = full dataset')
@click.option('--net_name', type=click.Choice   (['mnist_LeNet', 'fmnist_LeNet', 'cifar10_LeNet', 'arrhythmia_mlp',
                                               'cardio_mlp', 'satellite_mlp', 'satimage-2_mlp', 'shuttle_mlp',
                                               'thyroid_mlp', 'iiot_no_cat', 'iiot_emb', 'iiot_no_emb']),default='iiot_no_cat',
              help='''For the iiot dataset,the MLP network has different modes for categorical features are available: 
              iiot_no_cat - cat features are excluded, 
              iiot_emb - embeddings for cat features, 
              iiot_no_emb - cat features are not pre-prcoessed, treated like num features''')
@click.option('--net_h1', type=int, default=32, 
              help='''Number of the neurons of the first hidden layer
              The second layer is set to h1/2 and the last layer is h1/4''')
@click.option('--xp_path', type=click.Path(exists=True), default = '../log')
@click.option('--data_path', type=click.Path(exists=True), default = '../data')
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--eta', type=float, default=1.0, help='Deep  SAD hyperparameter eta (must be 0 < eta).')
@click.option('--ratio_known_normal', type=float, default=0.0,
              help='Ratio of known (labeled) normal training examples.')
@click.option('--ratio_known_outlier', type=float, default=0.01,
              help='Ratio of known (labeled) anomalous training examples.')
@click.option('--ratio_pollution', type=float, default=0.05,
              help='Pollution ratio of unlabeled training data with unknown (unlabeled) anomalies.')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for Deep SAD network training.')
@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for Deep SAD network training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=(25,), multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for Deep SAD objective.')
@click.option('--pretrain', type=bool, default=False,
              help='Pretrain neural network parameters via autoencoder.')
@click.option('--ae_optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
@click.option('--ae_lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--ae_n_epochs', type=int, default=50, help='Number of epochs to train autoencoder.')
@click.option('--ae_lr_milestone', type=int, default=(25,), multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
@click.option('--ae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--num_threads', type=int, default=0,
              help='Number of threads used for parallelizing CPU operations. 0 means that all resources are used.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
@click.option('--known_outlier_class', type=int, default=1,
              help='Specify the known outlier class of the dataset for semi-supervised anomaly detection.')
@click.option('--n_known_outlier_classes', type=int, default=1,
              help='Number of known outlier classes.'
                   'If 0, no anomalies are known.'
                   'If 1, outlier class as specified in --known_outlier_class option.'
                   'If > 1, the specified number of outlier classes will be sampled at random.')
def main(hp_tune, fl_mode, fl_num_rounds,fl_dataset_index, dataset_name, dataset_size,net_name,net_h1, xp_path, data_path, load_config, load_model, eta,
         ratio_known_normal, ratio_known_outlier, ratio_pollution, device, seed,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay,
         pretrain, ae_optimizer_name, ae_lr, ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay,
         num_threads, n_jobs_dataloader, normal_class, known_outlier_class, n_known_outlier_classes):
    """
    Deep SAD, a method for deep semi-supervised anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print paths
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info('Log file is %s' % log_file)
    logger.info('Computation device: %s' % device)
    
    if fl_mode != 'server':
        logger.info('Data path is %s' % data_path)
        logger.info('Export path is %s' % xp_path)

        if num_threads > 0:
            torch.set_num_threads(num_threads)
        logger.info('Number of threads: %d' % num_threads)
        logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

        # Print experimental setup
        logger.info('Net: %s' % net_name)
        logger.info('Dataset: %s' % dataset_name)
        logger.info('Normal class: %d' % normal_class)
        logger.info('Ratio of labeled normal train samples: %.2f' % ratio_known_normal)
        logger.info('Ratio of labeled anomalous samples: %.2f' % ratio_known_outlier)
        logger.info('Pollution ratio of unlabeled train data: %.2f' % ratio_pollution)
        if n_known_outlier_classes == 1:
            logger.info('Known anomaly class: %d' % known_outlier_class)
        else:
            logger.info('Number of known anomaly classes: %d' % n_known_outlier_classes)
        # Print model configuration
        logger.info('Eta-parameter: %.2f' % cfg.settings['eta'])
        
    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)


    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        torch.cuda.manual_seed(cfg.settings['seed'])
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % cfg.settings['seed'])


    #Hyperparameter tuning
    if hp_tune == True:
        if(dataset_name != 'iiot'):
            logger.info('Hyperparameter tuning is only implemented for the iiot dataset')
            return
        
        logger.info('Hyperparameter tuning runs with HyperOptSearch and ASHAScheduler')
        logger.info('Hyperparameter search space: learning rate [0.0001, 0.001, 0.01, 0.1]')
        logger.info('Hyperparameter search space: neurons of the 1st hidden layer [16,32,64,128]')
        logger.info('Hyperparameter search space: batch size [64, 128 , 256]')

        tune_config = {"lr" : tune.choice([0.0001, 0.001, 0.01, 0.1]),
                'h1' : tune.choice([16,32,64]),
                'bs': tune.choice([64,128,256]),
                #'data_path': data_path,
                #'fl_dataset_index':fl_dataset_index,
                #'dataset_size': dataset_size,
                #'net_name': net_name,
                #'random_state': np.random.RandomState(cfg.settings['seed']),
                #'device': device,
                #'normal_class': normal_class,
                #'known_outlier_class': known_outlier_class, 
                #'n_known_outlier_classes': n_known_outlier_classes,
                #'ratio_known_outlier':ratio_known_outlier, 
                #'ratio_pollution': 0.05,
                #'eta': cfg.settings['eta'],
                #'optimizer_name': cfg.settings['optimizer_name'],
                #'n_epochs': cfg.settings['n_epochs'], 
                #'lr_milestone': cfg.settings['lr_milestone'],
                #'weight_decay': cfg.settings['weight_decay'],
                #'n_jobs_dataloader':n_jobs_dataloader
                }
        searcher = HyperOptSearch()
        scheduler = ASHAScheduler()
        ray.init()
        analysis = tune.run(train_tune,config=tune_config,
                            metric="f_score", mode="max", num_samples=18,scheduler=scheduler, search_alg=searcher)
        logger.info(f'Best configuration found by RayTune: {analysis.get_best_config(metric="f_score", mode="max")}')
        return


    #Federated setting
    if fl_mode == 'client':
        if(dataset_name != 'iiot'):
            logger.info('Federated learning is only implemented for the iiot dataset')
            return
        logger.info('Federated mode: client')
        dataset = load_dataset('iiot', data_path,fl_dataset_index,dataset_size,net_name, normal_class, known_outlier_class, n_known_outlier_classes,
                           ratio_known_normal, ratio_known_outlier, ratio_pollution,
                           random_state=np.random.RandomState(cfg.settings['seed']))
                           
        deepSAD = DeepSAD(cfg.settings['eta'])
        deepSAD.set_network(net_name = net_name,h1=net_h1)
        client = FL_Client(deepSAD,dataset,cfg.settings, device,n_jobs_dataloader)
        fl.client.start_numpy_client("localhost:8080", client)
        return

    if fl_mode == 'server':
        strategy = fl.server.strategy.FedAvg(min_fit_clients=2,min_eval_clients=2,min_available_clients=2)
        if(dataset_name != 'iiot'):
            logger.info('Federated learning is only implemented for the iiot dataset')
            return
        logger.info('Federated mode: server')
        fl.server.start_server(server_address = 'localhost:8080',strategy=strategy,config={"num_rounds": fl_num_rounds})
        return


    # Load data
    dataset = load_dataset(dataset_name, data_path,fl_dataset_index,dataset_size,net_name, normal_class, known_outlier_class, n_known_outlier_classes,
                           ratio_known_normal, ratio_known_outlier, ratio_pollution,
                           random_state=np.random.RandomState(cfg.settings['seed']))

    # Log random sample of known anomaly classes if more than 1 class
    if n_known_outlier_classes > 1:
        logger.info('Known anomaly classes: %s' % (dataset.known_outlier_classes,))

    # Initialize DeepSAD model and set neural network phi
    deepSAD = DeepSAD(cfg.settings['eta'])
    deepSAD.set_network(net_name, h1=net_h1)


    # If specified, load Deep SAD model (center c, network weights, and possibly autoencoder weights)
    if load_model:
        deepSAD.load_model(model_path=load_model, load_ae=True, map_location=device)
        logger.info('Loading model from %s.' % load_model)

    logger.info('Pretraining: %s' % pretrain)
    if pretrain and net_name != 'iiot_emb':
        # Log pretraining details
        logger.info('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
        logger.info('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
        logger.info('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
        logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
        logger.info('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
        logger.info('Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])

        # Pretrain model on dataset (via autoencoder)
        deepSAD.pretrain(dataset,
                         h1 = net_h1,
                         optimizer_name=cfg.settings['ae_optimizer_name'],
                         lr=cfg.settings['ae_lr'],
                         n_epochs=cfg.settings['ae_n_epochs'],
                         lr_milestones=cfg.settings['ae_lr_milestone'],
                         batch_size=cfg.settings['ae_batch_size'],
                         weight_decay=cfg.settings['ae_weight_decay'],
                         device=device,
                         n_jobs_dataloader=n_jobs_dataloader)

        # Save pretraining results
        deepSAD.save_ae_results(export_json=xp_path + '/ae_results.json')

    # Log training details
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

    # Train model on dataset
    deepSAD.train(dataset,
                  optimizer_name=cfg.settings['optimizer_name'],
                  lr=cfg.settings['lr'],
                  n_epochs=cfg.settings['n_epochs'],
                  lr_milestones=cfg.settings['lr_milestone'],
                  batch_size=cfg.settings['batch_size'],
                  weight_decay=cfg.settings['weight_decay'],
                  device=device,
                  n_jobs_dataloader=n_jobs_dataloader)

    # Test model
    deepSAD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    # Save results, model, and configuration
    deepSAD.save_results(export_json=xp_path + '/results.json')
    if pretrain == True:
        deepSAD.save_model(export_model=xp_path + '/model.tar')
    cfg.save_config(export_json=xp_path + '/config.json')

    # Plot most anomalous and most normal test samples
    # indices, labels, scores = zip(*deepSAD.results['test_scores'])
    # indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    # idx_all_sorted = indices[np.argsort(scores)]  # from lowest to highest score
    # idx_normal_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # from lowest to highest score

    # if dataset_name in ('mnist', 'fmnist', 'cifar10'):

    #     if dataset_name in ('mnist', 'fmnist'):
    #         X_all_low = dataset.test_set.data[idx_all_sorted[:32], ...].unsqueeze(1)
    #         X_all_high = dataset.test_set.data[idx_all_sorted[-32:], ...].unsqueeze(1)
    #         X_normal_low = dataset.test_set.data[idx_normal_sorted[:32], ...].unsqueeze(1)
    #         X_normal_high = dataset.test_set.data[idx_normal_sorted[-32:], ...].unsqueeze(1)

    #     if dataset_name == 'cifar10':
    #         X_all_low = torch.tensor(np.transpose(dataset.test_set.data[idx_all_sorted[:32], ...], (0,3,1,2)))
    #         X_all_high = torch.tensor(np.transpose(dataset.test_set.data[idx_all_sorted[-32:], ...], (0,3,1,2)))
    #         X_normal_low = torch.tensor(np.transpose(dataset.test_set.data[idx_normal_sorted[:32], ...], (0,3,1,2)))
    #         X_normal_high = torch.tensor(np.transpose(dataset.test_set.data[idx_normal_sorted[-32:], ...], (0,3,1,2)))

    #     plot_images_grid(X_all_low, export_img=xp_path + '/all_low', padding=2)
    #     plot_images_grid(X_all_high, export_img=xp_path + '/all_high', padding=2)
    #     plot_images_grid(X_normal_low, export_img=xp_path + '/normals_low', padding=2)
    #     plot_images_grid(X_normal_high, export_img=xp_path + '/normals_high', padding=2)


if __name__ == '__main__':
    main()



