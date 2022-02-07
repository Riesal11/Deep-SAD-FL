from locale import normalize
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchvision.datasets.utils import download_url

import os
import torch
import numpy as np
import pandas as pd
import zipfile
import logging


class IIoTDataset(Dataset):

    url = 'https://www.cse.wustl.edu/~jain/iiot2/ftp/wustl_iiot_2021.zip'

    def __init__(self, root: str, dataset_name: str, train=True, random_state=None, download=True):
        super(Dataset, self).__init__()

        self.classes = [0, 1]

        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = Path(root)
        self.train = train  # training set or test set
        self.zip_file = self.root / 'wustl_iiot_2021.zip'
        self.csv_file = self.root / 'wustl_iiot_2021.csv'
        
        logger = logging.getLogger()

        print
        if download:
            self.download()
            self.unzip()

        df = pd.read_csv(self.csv_file)
        df = df.sample(n=100000, random_state = random_state)
        
        if train:
            logger.info(f'''Train Set\n{df['Target'].value_counts(normalize=True)}''')
        else:
            logger.info(f'''Test Set\n{df['Target'].value_counts(normalize=True)}''')

        df = df.drop(['StartTime', 'LastTime', 'SrcAddr', 'DstAddr', 'sIpId', 'dIpId', 'Traffic'], axis = 1)
        data = df.values
        X = data[:,:-1]
        y = data[:,-1]
        idx_norm = y == 0
        idx_out = y == 1

        # 60% data for training and 40% for testing; keep outlier ratio
        X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(X[idx_norm], y[idx_norm],
                                                                                test_size=0.4,
                                                                                random_state=random_state)
        X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(X[idx_out], y[idx_out],
                                                                            test_size=0.4,
                                                                            random_state=random_state)

        X_train = np.concatenate((X_train_norm, X_train_out))
        X_test = np.concatenate((X_test_norm, X_test_out))
        y_train = np.concatenate((y_train_norm, y_train_out))
        y_test = np.concatenate((y_test_norm, y_test_out))

        # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
        scaler = StandardScaler().fit(X_train)
        X_train_stand = scaler.transform(X_train)
        X_test_stand = scaler.transform(X_test)

        # Scale to range [0,1]
        minmax_scaler = MinMaxScaler().fit(X_train_stand)
        X_train_scaled = minmax_scaler.transform(X_train_stand)
        X_test_scaled = minmax_scaler.transform(X_test_stand)

        if self.train:
            self.data = torch.tensor(X_train_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.int64)
        else:
            self.data = torch.tensor(X_test_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_test, dtype=torch.int64)

        self.semi_targets = torch.zeros_like(self.targets)


    def __getitem__(self, index):
        """
        Args:
        index (int): Index
        Returns:
        tuple: (sample, target, semi_target, index)
        """
        sample, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])
        return sample, target, semi_target, index

    def __len__(self):
        return len(self.data)

    def _check_zip_exists(self):
        return os.path.exists(self.zip_file)
    
    def _check_csv_exists(self):
        return os.path.exists(self.csv_file)

    def download(self):
        if self._check_zip_exists():
            return
        # download file
        download_url(self.url, self.root, self.zip_file)
        print('Done!')
    
    def unzip(self):
        if self._check_csv_exists():
            return
        with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
            zip_ref.extractall(self.root)
        

        





        