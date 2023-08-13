# Deep SAD: A Method for Deep Semi-Supervised Anomaly Detection
Original framework: https://github.com/lukasruff/Deep-SAD-PyTorch
Deep SAD FL: Federated Learning enabled 


## Installation
This code is written in `Python 3.7` and requires the packages listed in `requirements.txt`.

Clone the repository to your machine and directory of choice:
```
https://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@github.com/veronikaBek/Deep-SAD-FL.git
```

To run the code, we recommend setting up a virtual environment, e.g. using `virtualenv` or `conda`:

### `virtualenv`
```
# pip install virtualenv
cd <path-to-Deep-SAD-PyTorch-directory>
virtualenv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### `conda`
```
cd <path-to-Deep-SAD-PyTorch-directory>
conda create --name myenv
source activate myenv
while read requirement; do conda install -n myenv --yes $requirement; done < requirements.txt
```


## Running experiments
The original framwork wwas changed and received the following additional capabilities:
1) Addtional iIOT dataset: https://www.cse.wustl.edu/~jain/iiot2/index.html
2) Hypertuning with RayTune for the iIOT dataset
3) Different modes for categorical features for the iIOT dataset: 
  a) Categorical features are removed from the dataset
  b) Categorical Embeddings
  c) Categorical features are treated as numerical features
4) Federated client and server modes
  
### Deep SAD
You can run Deep SAD experiments using the `main.py` script.
You must specify the seed value.

```
cd <path-to-Deep-SAD-PyTorch-directory>

# activate virtual environment
source myenv/bin/activate  # or 'source activate myenv' for conda

# create folders for experimental output
mkdir log/DeepSAD
mkdir log/DeepSAD/iiot_test

# change to source directory
cd src

# run experiment
python main.py --xp_path ../log/DeepSAD/mnist_test --data_path ../data --seed 1;
```

Default settings:
1) iIOt dataset
2) full dataset
3) categorical features are removed
4) Ratio of labeled normal samples in the training set: 0
5) Ration of labeled outlier samples in the training set: 0.01
6) Ration of outlier samples in the unlabled data in the training set : 0.05
7) No pretraining with the autoencoder
8) lr = 0.001
9) epochs = 50
10) lr_milestone = 25
11) batch size = 128
12) weight decay =0.5e-6


Have a look into `main.py` for all possible arguments and options.


## Running the hyperparmeter tuning with the Raytune

```
python main.py --hp_tune True --seed 1;
```

In addtion you can specify:
1) ration of labeled and unlabeled samples (--ratio_known_normal, --ratio_known_outlier, --ratio_pollution)
2) dataset size (--dataset_size, e.g --dataset_size 10000). If -1 then full dataset is used
3) if you want to use a specifc subset of the dataset, you can specify the starting index and the ending index is defined as starting index+dataset size
(--fl_dataset_index, e.g --fl_dataset_index 10)
4) Mode for categorical features (--net_name, you can choose from:
              iiot_no_cat - cat features are excluded, 
              iiot_emb - embeddings for cat features, 
              iiot_no_emb - cat features are not pre-prcoessed, treated like num features
5) n_epochs, lr_milestone, weight_decay, pretrain, optimizer_name
  
  
## Running centralized training:

```
python main.py --seed 1;
```
In addtion you can specify:
1) ration of labeled and unlabeled samples (--ratio_known_normal, --ratio_known_outlier, --ratio_pollution)
2) dataset size (--dataset_size, e.g --dataset_size 10000). If -1 then full dataset is used
3) if you want to use a specifc subset of the dataset, you can specify the starting index and the ending index is defined as starting index+dataset size
(--fl_dataset_index, e.g --fl_dataset_index 10)
4) Mode for categorical features (--net_name, you can choose from:
              iiot_no_cat - cat features are excluded, 
              iiot_emb - embeddings for cat features, 
              iiot_no_emb - cat features are not pre-prcoessed, treated like num features
5) Number of the neurons in the first hidden layer (--h1, e.g --h1 16). The second layer will be set to h1/2 and the outputlayer to h1/4
6) lr, batch_size, n_epochs, lr_milestone, weight_decay, pretrain, optimizer_name


  
## Running federated training:

First start the server instance in one terminal
```
python main.py --fl_mode server;
```
In addtion you can specify number of rounds (--fl_num_rounds, eg --fl_num_rounds 3)


Then, start a new terminal for each client (min 2 clients)
```
python main.py --fl_mode client --seed 1;
python main.py --fl_mode client --seed 3;
```

In addtion, for each client you can specify:
1) ration of labeled and unlabeled samples (--ratio_known_normal, --ratio_known_outlier, --ratio_pollution)
2) dataset size (--dataset_size, e.g --dataset_size 10000). If -1 then full dataset is used
3) if you want to use a specifc subset of the dataset, you can specify the starting index and the ending index is defined as starting index+dataset size
(--fl_dataset_index, e.g --fl_dataset_index 10)
4) Mode for categorical features (--net_name, you can choose from:
              iiot_no_cat - cat features are excluded, 
              iiot_emb - embeddings for cat features, 
              iiot_no_emb - cat features are not pre-prcoessed, treated like num features
5) Number of the neurons in the first hidden layer (--h1, e.g --h1 16). The second layer will be set to h1/2 and the outputlayer to h1/4
6) lr, batch_size, n_epochs, lr_milestone, weight_decay, pretrain, optimizer_name

```


## License
MIT
