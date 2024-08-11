# Multi layer of autoencoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import numpy as np
import json

from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
import json
import argparse
from metrics import *

def relative_l2_error(x, y):
    """
    root mean square error: square-root of sum of all (x_i-y_i)**2
    """
    assert(x.shape == y.shape)
    mse = np.sum((x-y)**2, axis = (1,2,3))
    temp = np.sum((x)**2, axis = (1,2,3))
    return np.sqrt(mse/temp), mse, temp, np.sqrt(np.sum(mse)/np.sum(temp))



class AELinear(nn.Module):
    
    def __init__(self, input_dim = 16*16, latent = 8 ,  h_size = None, activation = nn.LeakyReLU):
        super(AELinear, self).__init__()
        
        def model_size(model):
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_params

            
        self.latent_dim = latent
        
        if h_size is None:
            
            h_size = latent*2
        
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=h_size, bias = True),
            activation(True),
            nn.Linear(in_features=h_size, out_features=latent, bias = True),     # Max Pooling Layer 1
            activation(True),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent, out_features=h_size, bias = True),
            activation(True),
            nn.Linear(in_features=h_size, out_features=input_dim, bias = True),     # Max Pooling Layer 1
        )
        
        en_size = model_size(self.encoder)
        de_size = model_size(self.decoder)
        
        print("model size:", en_size+de_size, "decoder size:", en_size)
        
    def forward(self, x):
        
        encoded = self.encoder(x)
        self.latent = encoded
        decoded = self.decoder(encoded)
        return decoded
    
    
class AELinear(nn.Module):
    
    def __init__(self, input_dim = 16*16, latent = 8 ,  h_size = None, activation = nn.LeakyReLU):
        super(AELinear, self).__init__()
        
        def model_size(model):
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_params

            
        self.latent_dim = latent
        
        if h_size is None:
            
            h_size = latent*2
        
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=latent, bias = True),
            activation(True),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent, out_features=input_dim, bias = True),     # Max Pooling Layer 1
        )
        
        en_size = model_size(self.encoder)
        de_size = model_size(self.decoder)
        
        print("model size:", en_size+de_size, "decoder size:", en_size)
        
    def forward(self, x):
        
        encoded = self.encoder(x)
        self.latent = encoded
        decoded = self.decoder(encoded)
        return decoded
    
    


def train_autoencoder(model, dataset, criterion, optimizer):
    device = next(model.parameters()).device
    
    model.train()
    
    running_loss = 0.0
    batchs = len(dataset)
        
    for input_data in dataset:
        input_data = input_data.to(device)
        outputs = model(input_data)
        loss = criterion(outputs, input_data)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    loss = running_loss /batchs
    return loss


def evaluate_autoencoder(model, dataset, d_size = 1536):
    device = next(model.parameters()).device
    
    model.eval()
    batchs = len(dataset)
    all_result = []
    all_label = []
    
    # var_mean = dataset.dataset.var_mean
    # var_std = dataset.dataset.var_std
    
    with torch.no_grad():
        for input_data in dataset:
            input_data = input_data.to(device)
            outputs = model(input_data)
        
            all_result.append(outputs.detach().cpu().numpy())
            all_label.append(input_data.detach().cpu().numpy())
            
        all_result = np.concatenate(all_result,axis=0)[:,:d_size]
        all_label = np.concatenate(all_label,axis=0)[:,:d_size]
        
        # all_result = (all_result * var_std) + var_mean
        # all_label =  (all_label * var_std) + var_mean

        nrmse = relative_rmse_error_ornl(all_label, all_result)
        # _,_,_, l2_error = relative_l2_error(all_label, all_result)W

    return float(nrmse), all_result, all_label, float(-1)

def save_json(json_pth, data):
    # Load the existing JSON data
    if os.path.exists(json_pth):
        with open(json_pth, 'r') as json_file:
            existing_data = json.load(json_file)
        existing_data.update(data)
        data = existing_data
        
    # Save the updated data back to the JSON file
    with open(json_pth, 'w') as json_file:
        json.dump(data, json_file, indent=4)
        

        
if __name__=='__main__':
    
    shared_folder = "/home/xiao.li/ClimateModeling/shared_fdmodel"
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', type=int, default = 512)
    parser.add_argument('--dataset', type=str, default = "XGC")
    parser.add_argument('--e3sm_path', '-epth', type=str,
                        default = "%s/e3sm/dataset/bin"%(shared_folder))
    parser.add_argument('--xgc_path', '-xpth', type=str, 
                        default =  "%s/xgc/dataset/d3d_coarse_v2_400.npy"%(shared_folder))
    parser.add_argument('--s3d_path', '-spth', type=str, 
                        default =  "%s/s3d/dataset/S3D/1.dataSet/nHep58/input_150to200steps.npy"%(shared_folder))
    
    parser.add_argument('--save_path', type=str, default = "./models_single_layer")
    parser.add_argument('--num_workers',  type=int, default = 4)
    parser.add_argument('--save_latent',  type=int, default = 0)
    parser.add_argument('--epochs',  type=int, default = 500)
    parser.add_argument('--check_point',  type=int, default = 20)
    parser.add_argument('--model_name',  type=str, default = "AE")
    parser.add_argument('--device',  type=str, default = "cuda:0")
    
    parser.add_argument('--data_length',  type=int, default = -1)
    parser.add_argument('--input_size', type = int, default = 1536)  # 1536 = 6*16*16
    
    
    # These parameters are used for E3SM dataset
    parser.add_argument('--n_frame', type=int, default = 6)
    parser.add_argument('--patch_size', type=int, default = 16)
    parser.add_argument('--seq', type=int, default = 8)
    parser.add_argument('--var_mode', type=str, default = "VAR1")
    
    # XGC dataset
    parser.add_argument('--plane', type=int, default = -1)
    
    # S3D dataset
    parser.add_argument('--s3d_n_frame', type=int, default = 1)
    parser.add_argument('--s3d_patch_size', type=int, default = 5)

    
    args = parser.parse_args()
    
    args_dict = vars(args)
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    datasets = []
    
    if "E3SM" in args.dataset:
        from dataset import ClimatePatch
        
        patch_size = args.patch_size
        n_frame = args.n_frame
        seqs = [args.seq]
        var_mode = args.var_mode
        
        src_size = int(n_frame * patch_size**2)
        input_size =  src_size if args.input_size<=0 else args.input_size
        
        dataset_e3sm = ClimatePatch(data_path =args.e3sm_path, seqs = seqs,  
                               return_path = False, mode = var_mode, n_frame = n_frame,
                               patch_size = patch_size, flatten=True, pre_load=True, input_size = args.input_size)
        
        save_path = "%s/%s_%s_%s_bs_%d_nf_%d_ps_%d"%(args.save_path, args.model_name, args.dataset, 
                                  var_mode, args.batch_size, n_frame, patch_size)
        
        print("E3SM   %d instances,  source size: %d  data size:"%(len(dataset_e3sm), src_size), dataset_e3sm[0].shape)
        
        datasets.append(dataset_e3sm)
        
        
    if "XGC" in args.dataset:
        from xgc_dataset import XGC
        
        dataset_xgc = XGC(data_path = args.xgc_path, data_name = 'd3d', 
                      plane_idx = args.plane, normalize = True, flatten = True, input_size = args.input_size)
        
        save_path = "%s/%s_%s_bs_%d"%(args.save_path, args.model_name, args.dataset, args.batch_size)
        
        src_size = int(39*39)
        input_size = src_size if args.input_size<=0 else args.input_size
        
        print("XGC    %d instances, source size: %d  data size:"%(len(dataset_xgc), src_size), dataset_xgc[0].shape)
        
        datasets.append(dataset_xgc)
        
        
    if "S3D" in args.dataset:
        from s3d_dataset import S3D
        
        s3d_psize = args.s3d_patch_size
        s3d_nframe = args.s3d_n_frame
        
        dataset_s3d = S3D(data_path = args.s3d_path, patch_size = s3d_psize, n_frame = s3d_nframe ,
                          normalize = True, flatten = True, 
                          input_size = args.input_size, data_length = args.data_length)
        
        save_path = "%s/%s_%s_bs_%d"%(args.save_path, args.model_name, args.dataset, args.batch_size)
        
        src_size = int(s3d_psize**2*s3d_nframe*58)
        input_size = src_size if args.input_size<=0 else args.input_size
        
        print("S3D    %d instances, source size: %d  data size:"%(len(dataset_s3d), src_size), dataset_s3d[0].shape)
        
        datasets.append(dataset_s3d)
        
        
    n_set = len(datasets)
    
    if args.data_length>0:
        for ds in datasets:
            ds.dataset_length = args.data_length
    
    
    cat_datasets = ConcatDataset(datasets)
        
    device = args.device
    data_loader = DataLoader(cat_datasets, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    
    test_loaders = [DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers = 2) for dataset in datasets]

    l_sizes = [128, 64, 32, 16, 8]
    
    for latent_size in l_sizes:

        best_nrmse = [[-1, 1e10] for i in range(n_set)]
        best_l2 = [[-1, 1e10] for i in range(n_set)]
        nrmse_all = [{} for i in range(n_set)]
        l2_all = [{} for i in range(n_set)]

        # model = AECov3D_v2(1, latent_size, 1, base_channel = base_channel,  data_size = patch_size, n_frame= n_frame)
        h_size = 200
        model = AELinear( input_dim = input_size, latent = latent_size ,  h_size = h_size)
        model = model.to(device)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        scheduler = MultiStepLR(optimizer, milestones=[int(args.epochs/4*3), int(args.epochs/5*4)], gamma=0.1)
        
        json_path  = save_path + "_ls_%d.json"%(latent_size)
        model_path = save_path + "_ls_%d.pt"%(latent_size)
        

        args_dict["model_dims"] = [input_size, h_size, latent_size]
        save_json(json_path,{"arguments": args_dict })
        
        print("Training -- data size: %d  latent size: %d  cr: %.1f"%(input_size, latent_size, input_size/latent_size))
        print("Scheduler:",[int(args.epochs/4*3), int(args.epochs/5*4)])
        
        for epoch in tqdm(range(args.epochs)):
            
            loss = train_autoencoder(model, data_loader, criterion  = criterion, optimizer = optimizer)
            scheduler.step()
            
            if (epoch%args.check_point==0) or (epoch>args.epochs-10):
                
                for j, test_loader in enumerate(test_loaders):
                    data_name = datasets[j].dataset_name
                    d_size = datasets[j].src_size
                    nrmse, recons_data, original_data, l2_error = evaluate_autoencoder(model, test_loader, d_size)
                    
                    
                    if data_name == "S3D":
                        dim2 = s3d_nframe*s3d_psize*s3d_psize
                        y= recons_data.reshape([-1, 58, dim2]).transpose([1,0,2]).reshape([58, -1])
                        x = original_data.reshape([-1, 58, dim2]).transpose([1,0,2]).reshape([58, -1])
                        nrmse = float(mean_relative_rmse_error_ornl(x, y))
                    
                    nrmse_all[j]["%d"%(epoch)] = float(nrmse)
                    l2_all[j]["%d"%(epoch)] = float(l2_error)
                    

                    if nrmse<=best_nrmse[j][1]:
                        torch.save(model.state_dict(), model_path)
                        best_nrmse[j] = [epoch, nrmse]
                        best_l2[j] = [epoch, l2_error]


                    save_json(json_path,{data_name:{"NRMSE":nrmse_all[j], "L2":l2_all[j], 
                                                    "best_nrmse":best_nrmse[j][1],
                                                    "best_index":best_nrmse[j][0], 
                                                    "best_l2":best_l2[j][1],
                                                     }})

                    print(data_name, "NRMSE: %.8f || %.8f    L2:  %.8f || %.8f"%(nrmse, best_nrmse[j][1], l2_error, best_l2[j][1]), recons_data.shape)