from locale import normalize
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchvision.datasets.utils import download_url
from sklearn.preprocessing import LabelEncoder

import os
import torch
import numpy as np
import pandas as pd
import zipfile
import logging


class IIoTDataset(Dataset):

    url = 'https://www.cse.wustl.edu/~jain/iiot2/ftp/wustl_iiot_2021.zip'

    def __init__(self, root: str, dataset_name: str,fl_dataset_index=-1, dataset_size=-1,net_name='iiot_no_cat',train=True, random_state=None, download=True):
        super(Dataset, self).__init__()

        self.classes = [0, 1]

        # does not work with newer torch version, maybe fix?
        # if isinstance(root, torch._six.string_classes):
        #     root = os.path.expanduser(root)
        self.root = Path(root)
        self.train = train  # training set or test set
        self.zip_file = self.root / 'wustl_iiot_2021.zip'
        self.csv_file = self.root / 'wustl_iiot_2021.csv'
        logger = logging.getLogger()

        if download:
            self.download()
            self.unzip()

        df = pd.read_csv(self.csv_file)

        df = df[(df.values  == "normal")|(df.values  == "DoS" ) ]


        if dataset_size != -1:
            if fl_dataset_index == -1:
                logger.info(f'Random sampling from the dataset of the size {dataset_size}')
                df = df.sample(n=dataset_size, random_state = random_state)
            else:
                start = fl_dataset_index
                end = min(fl_dataset_index+dataset_size,len(df))
                logger.info(f'Sampling from {start} dataset index to {end} index')
                df = df[start:end]

        logger.info('Dataset size: %s' % len(df))
        logger.info(f'''Classes in the dataset: {df['Traffic'].unique()}''')
     
        df = df.drop(['StartTime', 'LastTime', 'SrcAddr', 'DstAddr', 'sIpId', 'dIpId', 'Traffic'], axis = 1)
        df = df[['Sport','Dport','Proto','Mean', 'SrcPkts', 'DstPkts', 'TotPkts', 'DstBytes',
            'SrcBytes', 'TotBytes', 'SrcLoad', 'DstLoad', 'Load', 'SrcRate',
            'DstRate', 'Rate', 'SrcLoss', 'DstLoss', 'Loss', 'pLoss', 'SrcJitter',
            'DstJitter', 'SIntPkt', 'DIntPkt', 'Dur', 'TcpRtt', 'IdleTime',
            'Sum', 'Min', 'Max', 'sDSb', 'sTtl', 'dTtl', 'SAppBytes', 'DAppBytes',
            'TotAppByte', 'SynAck', 'RunTime', 'sTos', 'SrcJitAct', 'DstJitAct',
            'Target']] #cat features in the first 3 columns, easier for cat and num feature separation
        
        label_encoders = {}

        if(net_name =='iiot_emb'):
            for cat_col in ['Sport','Dport','Proto']:
                label_encoders[cat_col] = LabelEncoder()
                df[cat_col] = label_encoders[cat_col].fit_transform(df[cat_col])
        
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

        #separate num and categ features, if cat_no_emb then cat features are viewed as num features
        if net_name=='iiot_no_cat' or net_name=='iiot_emb': 
            X_train_cat = X_train[:,0:3]
            X_train = X_train[:,3:] #numerical features
            X_test_cat = X_test[:,0:3]
            X_test = X_test[:,3:] #numerical features

        # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
        #only num features
        scaler = StandardScaler().fit(X_train)
        X_train_stand = scaler.transform(X_train)
        X_test_stand = scaler.transform(X_test)

        # Scale to range [0,1]
        #only num features
        minmax_scaler = MinMaxScaler().fit(X_train_stand)
        X_train_scaled = minmax_scaler.transform(X_train_stand)
        X_test_scaled = minmax_scaler.transform(X_test_stand)

        if net_name=='iiot_emb': 
            X_train_scaled = np.concatenate((X_train_cat,X_train_scaled),axis=1)
            X_test_scaled = np.concatenate((X_test_cat,X_test_scaled),axis=1)

        if self.train:
            self.data = torch.tensor(X_train_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.int64)

        else:
            logger.info(f'{len(X_test_scaled[y_test==0])} normal samples in the testset')
            logger.info(f'{len(X_test_scaled[y_test==1])} outlier samples in the testset')
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
        

        





        