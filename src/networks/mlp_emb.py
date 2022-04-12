from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from base.base_net import BaseNet


class MLP_emb(BaseNet):

    def __init__(self, x_dim, h_dims=[128, 64], rep_dim=32, emb_dims=[(51057, 50), (7782, 50), (8, 4)], cont_dims = 38, bias=False):
        super().__init__()

        #embedding layer
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        self.emb_dims= sum([y for x, y in emb_dims]) #dimensions of the new embedding layer

        #batch normalization for the continuous features
        self.cont_dims = cont_dims
        self.first_bn_layer = nn.BatchNorm1d(self.cont_dims)

        neurons = [self.emb_dims + self.cont_dims, *h_dims]  #new mlp input layer: vector dim of the embeddings + num of cont. features     
        layers = [Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=bias) for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(layers)

        #output layers
        self.rep_dim = rep_dim
        self.code = nn.Linear(h_dims[-1], rep_dim, bias=bias)

    def forward(self, x):
        x_cat = x[:,0:3]    
        x_cont = x[:,3:]
        x_cat = x_cat.clone().detach().to(torch.int64)

        x_cat = [emb_layer(input=x_cat[:, i]) for i,emb_layer in enumerate(self.emb_layers)]
        x_cat = torch.cat(x_cat, 1)

        normalized_cont_data = self.first_bn_layer(x_cont)

        x = torch.cat([x_cat, normalized_cont_data], 1) 

        
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)
        return self.code(x)


class Linear_BN_leakyReLU(nn.Module):
    """
    A nn.Module that consists of a Linear layer followed by BatchNorm1d and a leaky ReLu activation
    """

    def __init__(self, in_features, out_features, bias=False, eps=1e-04):
        super(Linear_BN_leakyReLU, self).__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features, eps=eps, affine=bias)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.linear(x)))
