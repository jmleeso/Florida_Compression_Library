import torch
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
import numpy as np
import json
from tqdm import tqdm
import threading


def create_patches(data, patch_size, n_frame):
    frame = data.shape[1]//n_frame
    length = data.shape[-1]//patch_size
    num_patches = frame*length*length
    patches = []
    
    for k in range(frame):
        for i in range(length):
            for j in range(length):
                z = k*n_frame
                x = i*patch_size
                y = j*patch_size
                patch = data[:, z:z+n_frame,x:x+patch_size,y:y+patch_size]
                patches.append(patch)
                
    patches = np.asarray(patches)
    return patches

class S3D(Dataset):
    def __init__(self, data_path, patch_size = 5, n_frame = 1 ,normalize = True, flatten = False, input_size = -1, data_length = -1):
        
        self.dataset_name = "S3D"
        self.src_size = n_frame*patch_size**2*58
        data = np.load(data_path)[:,0:6, ]
        self.data = create_patches(data, patch_size, n_frame)
        
        if data_length>0:
            self.data = self.data[:data_length]

        self.flatten = flatten
        self.input_size = input_size

            
        self.var_mean = np.mean(self.data, axis = (0,2,3,4), keepdims = True)
        self.var_std = np.std(self.data, axis = (0,2,3,4), keepdims = True)
        
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
    

