from email.policy import default
import time
import click
import torch
import logging
import random
import numpy as np

import flwr as fl
from flwr.common import Context
from flwr.client import Client
from Client import FL_Client


import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from tune import train_tune

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from DeepSAD import DeepSAD
from datasets.main import load_dataset

from flower_async.async_server import AsyncServer
from flower_async.async_strategy import AsynchronousStrategy
from flower_async.async_client_manager import AsyncClientManager

import binascii
import csv
from threading import Event, Thread, Timer
from kafka import KafkaConsumer, KafkaProducer

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
@click.option('--n_epochs', type=int, default=5, help='Number of epochs to train.')     # initial: 50
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
@click.option('--ae_n_epochs', type=int, default=5, help='Number of epochs to train autoencoder.') # initial: 50
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
@click.option('--server_ip_address', type=str, default="127.0.0.1:8080",
              help='The ip address of the server.')
@click.option('--download_zip', type=bool, default=False,
              help='Specify if IIoT Dataset should be downloaded if not present')
@click.option('--client_id', type=int, default=1,
              help='Specify the ID of the client')
def main(hp_tune, fl_mode, fl_num_rounds,fl_dataset_index, dataset_name, dataset_size,net_name,net_h1, xp_path, data_path, load_config, load_model, eta,
         ratio_known_normal, ratio_known_outlier, ratio_pollution, device, seed,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay,
         pretrain, ae_optimizer_name, ae_lr, ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay,
         num_threads, n_jobs_dataloader, normal_class, known_outlier_class, n_known_outlier_classes, server_ip_address, download_zip, client_id):
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
        logger.info('Hyperparameter search space: learning rate [0.0001, 0.001, 0.01]')
        logger.info('Hyperparameter search space: neurons of the 1st hidden layer[64,128,256,512]')
        logger.info('Hyperparameter search space: batch size [64,128,256]')

        tune_config = {"lr" : tune.choice([0.0001, 0.001, 0.01]),
                'h1' : tune.choice([64,128,256,512]),
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
        ray.init(num_cpus=2,num_gpus=1)
        analysis = tune.run(train_tune,config=tune_config,resources_per_trial={'gpu':1,'cpu':2},
                            metric="roc_auc", mode="max", num_samples=18,scheduler=scheduler, search_alg=searcher)
        logger.info(f'Best configuration found by RayTune: {analysis.get_best_config(metric="roc_auc", mode="max")}')
        return
    

    class PollingThread(Thread):
        def __init__(self, event, *args):
            Thread.__init__(self)
            self.stopped = event
            self.consumer: KafkaConsumer = args[0]
            self.client_id = str(args[1])
            print("client id: " + self.client_id)

        def run(self):
            while not self.stopped.wait(5.0):
                print("my thread")
                self.poll_distributor()

        def poll_distributor(self):
            records = self.consumer.poll(10.0)
            filename = "../data/wustl_iiot_2021.csv"

            with open(filename, "a") as f:
                writer = csv.writer(f, delimiter=",", lineterminator='\r')
                for topic_data, consumer_records in records.items():
                    if topic_data.topic == "client-"+self.client_id:
                        for consumer_record in consumer_records:
                            row = consumer_record.value.split(',')
                            writer.writerow(row)

    def client_fn(context: Context) -> Client:
        """Create a Flower client representing a single organization."""
        
        # Load model
        deepSAD = DeepSAD(xp_path,cfg.settings['eta'])
        deepSAD.set_network(net_name = net_name,h1=net_h1)

        # Load data
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data partition
        # Read the node_config to fetch data partition associated to this node
        # partition_id = context.node_config["partition-id"]
        logger.info("loading dataset...")
        dataset = load_dataset('iiot', data_path,fl_dataset_index,dataset_size,net_name, normal_class, known_outlier_class, n_known_outlier_classes,
                           ratio_known_normal, ratio_known_outlier, ratio_pollution,
                           random_state=cfg.settings['seed'], download_zip=download_zip)

        # Create a single Flower client representing a single organization
        return FL_Client(deepSAD,dataset,cfg.settings, device,n_jobs_dataloader).to_client()

               

    #Federated setting

    if fl_mode == 'client':
        if(dataset_name != 'iiot'):
            logger.info('Federated learning is only implemented for the iiot dataset')
            return
        logger.info('Federated mode: client')
        logger.info('Client id %s', client_id)
        # with single client
        # dataset = load_dataset('iiot', data_path,fl_dataset_index,dataset_size,net_name, normal_class, known_outlier_class, n_known_outlier_classes,
        #                    ratio_known_normal, ratio_known_outlier, ratio_pollution,
        #                    random_state=cfg.settings['seed'])
        # deepSAD = DeepSAD(xp_path,cfg.settings['eta'])
        # deepSAD.set_network(net_name = net_name,h1=net_h1)
        # client = FL_Client(deepSAD,dataset,cfg.settings, device,n_jobs_dataloader).to_client()
        # fl.client.start_client(server_address = server_ip_address, client= client)


        # use kafka:9092 in container or localhost:29092 on host
        # value_serializer=lambda v: binascii.hexlify(v.encode('utf-8')), 
        producer = KafkaProducer(value_serializer=lambda v: binascii.hexlify(v.encode('utf-8')),
                                 key_serializer=lambda k: binascii.hexlify(k.encode('utf-8')),
                                 bootstrap_servers='10.0.0.20:29092')
        producer.send('distributor', key="new-client", value=str(client_id))


        consumer = KafkaConsumer(bootstrap_servers='10.0.0.20:29092',
            value_deserializer=lambda v: binascii.unhexlify(v).decode('utf-8'),
            key_deserializer=lambda k: binascii.unhexlify(k).decode('utf-8'),
            client_id=client_id,
            consumer_timeout_ms=1000)
        client_topic = "client-"+ str(client_id)
        while client_topic not in consumer.topics():
            logger.info(consumer.topics())
            time.sleep(1)
        logger.info("new client topic ready to go!")
        consumer.subscribe([client_topic])
        stopFlag = Event()
        thread = PollingThread(stopFlag, consumer, client_id)
        thread.start()


        # this will stop the timer
        # stopFlag.set()

        # with client_fn
        fl.client.start_client(server_address = server_ip_address, client_fn= client_fn)
        return

    if fl_mode == 'server':
        # initial sync
        # strategy = fl.server.strategy.FedAvg(min_fit_clients=2,min_evaluate_clients=2,min_available_clients=2)
        
        # async test
        server = AsyncServer(
            base_conf_dict=dict(),
            strategy=fl.server.strategy.FedAvg(min_fit_clients=2,min_evaluate_clients=2,min_available_clients=2), 
            # client_manager=fl.server.SimpleClientManager(), 
            client_manager=AsyncClientManager(),
            async_strategy=AsynchronousStrategy(async_aggregation_strategy='fedasync', fedasync_a=1.0,total_samples=1000000, staleness_alpha=1.0, fedasync_mixing_alpha=1.0, num_clients=2, use_staleness=False, use_sample_weighing=False, send_gradients=False, server_artificial_delay=False))
        
        config = fl.server.ServerConfig(num_rounds=5)
        if(dataset_name != 'iiot'):
            logger.info('Federated learning is only implemented for the iiot dataset')
            return
        logger.info('Federated mode: server')

        # initial sync
        # fl.server.start_server(server_address = server_ip_address,strategy=strategy,config=config)

        # async test
        fl.server.start_server(server=server, server_address = server_ip_address,strategy=server.strategy,config=config)
        return


    ## TODO: REMOVE

    # Load data
    dataset = load_dataset(dataset_name, data_path,fl_dataset_index,dataset_size,net_name, normal_class, known_outlier_class, n_known_outlier_classes,
                           ratio_known_normal, ratio_known_outlier, ratio_pollution,
                           random_state=cfg.settings['seed'])

    # Log random sample of known anomaly classes if more than 1 class
    if n_known_outlier_classes > 1:
        logger.info('Known anomaly classes: %s' % (dataset.known_outlier_classes,))

    # Initialize DeepSAD model and set neural network phi
    deepSAD = DeepSAD(xp_path,cfg.settings['eta'])
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



