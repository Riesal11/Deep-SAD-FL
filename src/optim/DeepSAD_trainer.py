from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
from utils import write_results_to_csv


import logging
import time
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import tkinter

class DeepSADTrainer(BaseTrainer):

    def __init__(self, c, eta: float,log_file: str, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader,log_file)

        # Deep SAD parameters
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.eta = eta

        # Optimization parameters
        self.eps = 1e-6

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None
        self.test_loss = None
        self.test_f1 = None
        self.test_precision = None
        self.test_recall = None
        self.test_precision_norm = None
        self.test_recall_norm = None
        self.test_f1_norm = None
        self.log_file = log_file

    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized. c = %s' % self.c)

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, semi_targets, _ = data
                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                loss = torch.mean(losses)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet, use_full_dataset: bool = False):
        logger = logging.getLogger()

        if use_full_dataset:
            # special case where the full dataset is tested
            logger.info('Using full dataset for testing...')
            _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader, use_full_dataset=use_full_dataset)
        else:
            # Get test data loader
            _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Due to client_fn in main, c is not defined here, so I do the same again
        # TODO: check if this has any complications
        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(test_loader, net)
            logger.info('Center c initialized. c = %s' % self.c)
        
        # Set device for network
        net = net.to(self.device)

        # Testing
        logger.info('Starting testing...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, semi_targets, idx = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                semi_targets = semi_targets.to(self.device)
                idx = idx.to(self.device)

                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                loss = torch.mean(losses)
                scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)      
        self.test_auc = roc_auc_score(labels, scores)
        fpr, tpr, roc_threshold = roc_curve(labels, scores)
        plt.figure()
        plt.title('ROC Curve')
        plt.plot(fpr, tpr, 'b', label = 'AUC-ROC = {:.2f}%'.format(100. * self.test_auc),linewidth=5)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        today = datetime.now()
        date_time_1 = today.strftime("%H_%M_%S_%f")
        print("date and time 1:",date_time_1)
        plt.savefig(self.log_file + '/auc_roc' + str(date_time_1) + '.png')
        plt.close()

        plt.figure()
        precision, recall, threshold = precision_recall_curve(labels, scores)
        plt.title('PR Curve')
        plt.plot(recall,precision, 'b',label= 'AUC-PR = {:.2f}%'.format(100. * auc(recall,precision)),linewidth=5)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        date_time_2 = today.strftime("%H_%M_%S_%f")
        print("date and time 1:",date_time_2)
        plt.savefig(self.log_file + '/auc_pr' + str(date_time_1) + '.png')



        #selecting the threshold that gives the highest f1-score
        np.seterr(invalid='ignore')
        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.nanargmax(fscore)
        threshold_opt = threshold[ix]

        #classifying the anomaly scores based on the selected threshold
        y_pred = np.where(scores>=threshold_opt, 1, 0)

        #metrics for the anomaly class
        test_precision, test_recall, test_f1, _ =  precision_recall_fscore_support(labels, y_pred, average='binary')
        #metrics for the anomaly class
        test_precision_norm, test_recall_norm, test_f1_norm, _ =  precision_recall_fscore_support(labels, y_pred, pos_label=0, average='binary')


        self.test_f1 = test_f1
        self.test_precision = test_precision
        self.test_recall = test_recall

        self.test_precision_norm = test_precision_norm
        self.test_recall_norm = test_recall_norm
        self.test_f1_norm = test_f1_norm

        self.test_loss = epoch_loss / n_batches

        # Log results
        logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info(f'Anomaly scores range from {min(scores)} to {max(scores)}')
        logger.info(f'Best Threshold {threshold_opt} with the F1-score {test_f1}')
        logger.info(f'Precision {test_precision}')
        logger.info(f'Recall {test_recall}')
        logger.info(f'Precision of the normal class {test_precision_norm}')
        logger.info(f'Recall of the normal class {test_recall_norm}')
        logger.info(f'F1-score of the normal class {test_f1_norm}')
        logger.info('Test PR-AUC: {:.2f}%'.format(100. * auc(recall,precision)))
        logger.info('Test ROC-AUC: {:.2f}%'.format(100. * self.test_auc))
        logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Finished testing.')

        write_results_to_csv(file_path=self.log_file + '/test_results.csv', 
                            test_loss=self.test_loss, 
                            anomaly_scores_min=min(scores),
                            anomaly_scores_max=max(scores),
                            best_threshold_f1=f'{threshold_opt}, {test_f1}',
                            test_precision=test_precision,
                            test_recall=test_recall,
                            test_precision_norm=test_precision_norm,
                            test_recall_norm=test_recall_norm,
                            test_f1_norm=test_f1_norm,
                            test_pr_auc=100. * auc(recall,precision),
                            test_roc_auc=100. * self.test_auc,
                            test_time=self.test_time
                            )
        
    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
