from .mnist import MNIST_Dataset
from .fmnist import FashionMNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .odds import ODDSADDataset
from .iiot import IIOTADDataset


def load_dataset(dataset_name, data_path,fl_dataset_index,dataset_size,net_name,normal_class, known_outlier_class, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0,
                 random_state=None, download_zip=False):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'fmnist', 'cifar10',
                            'arrhythmia', 'cardio', 'satellite', 'satimage-2', 'shuttle', 'thyroid', 'iiot')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path,
                                normal_class=normal_class,
                                known_outlier_class=known_outlier_class,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution)

    if dataset_name == 'fmnist':
        dataset = FashionMNIST_Dataset(root=data_path,
                                       normal_class=normal_class,
                                       known_outlier_class=known_outlier_class,
                                       n_known_outlier_classes=n_known_outlier_classes,
                                       ratio_known_normal=ratio_known_normal,
                                       ratio_known_outlier=ratio_known_outlier,
                                       ratio_pollution=ratio_pollution)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path,
                                  normal_class=normal_class,
                                  known_outlier_class=known_outlier_class,
                                  n_known_outlier_classes=n_known_outlier_classes,
                                  ratio_known_normal=ratio_known_normal,
                                  ratio_known_outlier=ratio_known_outlier,
                                  ratio_pollution=ratio_pollution)

    if dataset_name in ('arrhythmia', 'cardio', 'satellite', 'satimage-2', 'shuttle', 'thyroid'):
        dataset = ODDSADDataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)

    if dataset_name == 'iiot':
        dataset = IIOTADDataset(root=data_path,
                            dataset_name=dataset_name,
                            fl_dataset_index=fl_dataset_index,
                            dataset_size=dataset_size,
                            net_name=net_name,
                            n_known_outlier_classes=n_known_outlier_classes,
                            ratio_known_normal=ratio_known_normal,
                            ratio_known_outlier=ratio_known_outlier,
                            ratio_pollution=ratio_pollution,
                            random_state=random_state,
                            download_zip=download_zip)

    return dataset
