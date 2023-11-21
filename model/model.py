import torch
import torch.nn as nn  
import torch.nn.functional as F
import numpy as np 


class CNN(nn.Module):

    def __init__(self,
                 # Conv params 
                 n_in:int, n_out:int,n_hidden:int, k:int, 
                 # MLP params 
                 fc_hidden: int, n_classes:int,
                 bn = True)->None:

        # If we use Batch Normalization or not 
        self.batch_n = bn
        
        # Convolutional Layers and Batch Normalization Layers 

        # Initial Layer  
        self.conv_l, self.bn_l = [], []
        self.conv_l.append(nn.Conv2d(n_in,n_hidden,3)) 
        self.bn_l.append(nn.BatchNorm1d(n_hidden)) 

        # Hidden layers 
        for _ in range(k-2):
            self.conv_l.append(nn.Conv2d(n_hidden,n_hidden,3)) 
            self.bn_l.append(nn.BatchNorm1d(n_hidden)) 

        # Last Layer with n_out 
        self.conv_l.append(nn.Conv2d(n_hidden,n_out,3)) 
        self.bn_l.append(nn.BatchNorm1d(n_out)) 

        # Transform to ModuleList
        self.conv_l = nn.ModuleList(self.conv_l)
        self.bn_l= nn.ModuleList(self.bn_l)
        

        # Fully Connected Part 
        # We just use one single Sequential
        self.fc_layers = nn.Sequential(
                nn.Flatten(), 
                nn.Linear(n_out,fc_hidden),
                nn.ReLU(), 
                nn.Linear(fc_hidden,fc_hidden), 
                nn.ReLU(),   
                nn.Linear(fc_hidden,n_classes), 
                nn.ReLU(),   
        ) 

        # Batch Norm Inizialization 
        if self.batch_n: 
            for m in self.modules():
                if isinstance(m, (torch.nn.BatchNorm1d)):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self,x):
        # Convolutional Part 
        for conv,bn in (self.conv_l,self.bn_l):
            x = conv() 
            if self.batch_n: x = bn(x)
            x = F.relu(x)
        # Fully connected Output 
        return nn.functional.softmax(self.fc_layers(x))

    def loss(self,x: torch.Tensor,y: torch.Tensor)-> torch.Tensor:
        return F.cross_entropy(x,y) 


