import torch
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
import numpy as np
import json
from tqdm import tqdm
import threading



class XGC(Dataset):
    def __init__(self, data_path, data_name = 'd3d', plane_idx = 0, normalize = True, flatten = False, input_size = -1):
        
        self.src_size = 39*39
        
        self.dataset_name = "XGC"
        self.flatten = flatten
        self.input_size = input_size
        
        self.data = np.load(data_path)
        
        if data_name == 'd3d':
            self.data = self.data.reshape([8, -1, 1, 39, 39])
            
        self.data  = self.data[plane_idx] if plane_idx>=0 else self.data.reshape([-1, 1, 39, 39])

            
        self.var_mean = np.mean(self.data)
        self.var_std = np.std(self.data)
        
        if normalize:
            self.data = (self.data - self.var_mean)/self.var_std
            
        self.dataset_length = self.data.shape[0]

        
    def __len__(self):
        return self.dataset_length
    
        

    def __getitem__(self, idx):
        
        if self.flatten:
            input_data = torch.FloatTensor(self.data[idx].reshape(-1))
            
            if self.input_size>0:
                n_zeros = self.input_size - input_data.shape[0]
                input_data = torch.nn.functional.pad(input_data, (0, n_zeros))
                
            return input_data
    
    
        return torch.FloatTensor(self.data[idx])
    

