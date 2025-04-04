from torch.utils.data import DataLoader, Subset, ConcatDataset
from base.base_dataset import BaseADDataset
from base.iiot_dataset import IIoTDataset
from .preprocessing import create_semisupervised_setting



import torch


class IIOTADDataset(BaseADDataset):

    def __init__(self, root: str, dataset_name: str,fl_dataset_index:int = -1, dataset_size: int = -1,net_name: str = 'iiot_no_cat', n_known_outlier_classes: int = 0, ratio_known_normal: float = 0.0,
                 ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0, random_state=None, download_zip: bool=False):
        super().__init__(root)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = (0,)
        self.outlier_classes = (1,)

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        else:
            self.known_outlier_classes = (1,)

        # Get train set
        train_set = IIoTDataset(root=self.root, dataset_name=dataset_name,fl_dataset_index=fl_dataset_index,dataset_size=dataset_size,net_name=net_name,train=True, random_state=random_state, download_zip=download_zip)

        # Create semi-supervised setting
        idx, _, semi_targets = create_semisupervised_setting(train_set.targets.cpu().data.numpy(), self.normal_classes,
                                                             self.outlier_classes, self.known_outlier_classes,
                                                             ratio_known_normal, ratio_known_outlier, ratio_pollution)
        train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels

        # Subset train_set to semi-supervised setup
        self.train_set = Subset(train_set, idx)

        # Get test set
        self.test_set = IIoTDataset(root=self.root, dataset_name=dataset_name,fl_dataset_index=fl_dataset_index,dataset_size=dataset_size,net_name=net_name, train=False, random_state=random_state)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0, use_full_dataset: bool = False) -> (
            DataLoader, DataLoader):
        if use_full_dataset:
            full_dataset = ConcatDataset([self.train_set, self.test_set])
            full_loader = DataLoader(dataset=full_dataset, batch_size=batch_size, shuffle=shuffle_test,
                                  num_workers=num_workers, drop_last=True)
            return full_loader, full_loader
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)
        return train_loader, test_loader
