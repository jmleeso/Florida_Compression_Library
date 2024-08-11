import torch
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
import numpy as np
import json
from tqdm import tqdm
import threading

def run_decorrelate(data):

    row_cat  = np.concatenate([d for d in data],axis=0)
    col_cat  = np.concatenate([d for d in data],axis=1)
    U1, S1, Vh1 = np.linalg.svd(row_cat.T)  # shape = (8, 8) (8,) (160, 160)
    U2, S2, Vh2 = np.linalg.svd(col_cat)   # shape = (8, 8) (8,) (160, 160)

    decorrelated_data = (U1.T@data@U2)
    return decorrelated_data, U1, U2

def run_parallel(function, data_list, output_list, indexes, thread_id = 0):
    
    indexes =  tqdm(indexes) if thread_id==0 else indexes 
    for idx in indexes:
        output_list[idx] = function(data_list(idx))



class ClimatePatch(Dataset):
    def __init__(self, data_path, label_path = None, seqs=[0], 
                 return_path = False, mode = "TC", patch_size = 40, n_frame = 1, 
                 flatten= False, pre_load = False, extra_dim = False, 
                 return_index = False, decorrelate_frame = False, norm_each = False, 
                 normalize = True, overlap_step = False, input_size = -1):
        
        
        self.dataset_name = "E3SM"
        self.data_path = data_path
        self.var5_path = data_path.replace("dataset_rain", "dataset_numpy")
        self.IVT_path = data_path.replace("dataset_numpy", "dataset_rain")
        
        self.mode = mode
        self.patch_size = patch_size
        self.overlap_step = overlap_step
        self.input_size = input_size
        
        self.src_size = n_frame*patch_size**2
        
        if self.overlap_step<=0:
            self.side_length = int(240//self.patch_size)
            self.cut_step = self.patch_size
        else:
            assert(patch_size%self.overlap_step == 0)
            self.side_length = int((240 - self.patch_size)/self.overlap_step + 1)
            self.cut_step = self.overlap_step
            
        self.n_patch = int(self.side_length *self.side_length)    
        self.flatten =flatten
        self.pre_load = pre_load
        self.n_frame = n_frame
        self.extra_dim = extra_dim
        self.return_index = return_index
        self.decorrelate_frame = decorrelate_frame
        self.normalize = normalize
        
        
        self.norm_each = norm_each
        
        print("Dataset normalization:", normalize)
        print("Samples per side:", self.n_patch)
        
        
        
        # assert(240==(self.side_length*self.patch_size))

        
        self.n_side = 6
        self.output_size = 240
        
        self.return_path = return_path
        
        self.all_paths = []
        self.all_var5_paths = []
        self.all_IVT_paths = []
        
        self.ar_maskpath = []
        self.label_idx = []
        
        
        if label_path is None:
            label_path = data_path
        
        for seq in seqs:
            
            self.all_var5_paths.extend(sorted(glob(os.path.join(self.var5_path,"%02d"%seq, "*.bin"))))
            self.all_IVT_paths.extend(sorted(glob(os.path.join(self.IVT_path,"%02d"%seq, "*.bin"))))
            
            
            self.ar_maskpath.extend(sorted(glob(os.path.join(label_path,"ar_mask","%02d"%seq, "*.bin"))))
            
            self.label_idx.extend([i+720*seq for i in range(720)])
        
        if "AR" in mode[:2]:
            self.all_paths = self.all_IVT_paths
            print("load variable for AR")
        else:
            self.all_paths = self.all_var5_paths
            print("load variable for TC")
        
        self.all_data = []
        
        if self.pre_load:
            for i in range(len(self.all_paths)):
                temp_data = np.fromfile(self.all_paths[i], dtype = np.float32).reshape([-1, 6, 240, 240])
                
                if mode == "TC1":
                    temp_data = temp_data[0:1]
                elif "VAR" in mode:
                    idx = int(self.mode[-1])
                    temp_data = temp_data[idx-1:idx]
                elif "SELECT" in mode:
                    idx = [int(v)-1 for v in mode.replace("SELECT","")]
                    temp_data = temp_data[idx]
                elif "AR" in mode:
                    idx = [int(v)-1 for v in mode.replace("AR","")]
                    temp_data = temp_data[idx]
                       
                self.all_data.append(temp_data[None])
            
            self.all_data = np.concatenate(self.all_data, axis = 0)
            self.var_mean = np.mean(self.all_data, axis = (0,2,3,4), keepdims = True)
            self.var_std  = np.std(self.all_data, axis = (0,2,3,4), keepdims = True)
            
            self.x_max = np.max(self.all_data, axis = (0,2,3,4), keepdims = True)
            self.x_min = np.min(self.all_data, axis = (0,2,3,4), keepdims = True)
            
            
        print("self.all_data", self.all_data.shape)    
        print(len(self.all_paths), len(self.ar_maskpath), self.all_data.shape, self.var_mean.shape)
        

        if self.decorrelate_frame:
            self.all_decorrelated_data, self.all_u1, self.all_u2 = self.decorrelate()
            self.all_u1 = np.concatenate(self.all_u1, axis=0).astype(np.float32)
            self.all_u2 = np.concatenate(self.all_u2, axis=0).astype(np.float32)
            
            self.all_u1 = torch.from_numpy(self.all_u1)[:, None]
            self.all_u2 = torch.from_numpy(self.all_u2)[:, None]
        
        if self.norm_each:
            self.calculate_mean_var()
            
            
        self.update_length()
            
    def update_length(self):
        self.dataset_length = int(len(self.all_paths)*self.n_side*self.n_patch/self.n_frame)
        return self.dataset_length
        
    def calculate_mean_var(self,):
        self.inst_means = {}
        self.inst_stds = {}
        
        for i in tqdm(range(self.__len__())):
            inst_data = self.cutting_frame(i, normalize =False)
            self.inst_means[i], self.inst_stds[i] = np.mean(inst_data), np.std(inst_data)
            
        
        
    def decorrelate(self, n_core = 6):
        self.all_decorrelated_data = [] 
        self.all_u1 = []
        self.all_u2 = []
        
        print("decorrelate frame")
        n_samples = self.__len__()
        
        # n_samples = 1000
        
        n_percore = n_samples//n_core
        
        process_results = [None for i in range(n_samples)]
        all_thread = []
        for n in range(n_core):
            if n<(n_core-1):
                indexes = range(n*n_percore,(n+1)*n_percore)
            else:
                indexes = range(n*n_percore, n_samples)
            
            print("start thread:", n, indexes)
            thread1 = threading.Thread(target = run_parallel, args = (run_decorrelate, self.cutting_frame, process_results, indexes, n,))
            thread1.start()
            
            all_thread.append(thread1)
        
        for thread in all_thread:
            thread.join()
        
        
        for decorrelated_data, U1, U2 in process_results:
            self.all_decorrelated_data.append(decorrelated_data.astype(np.float32))
            self.all_u1.append(U1[None])
            self.all_u2.append(U2[None])
            
        
#         for index in tqdm(range()):
            
#             inst_data = self.cutting_frame(index)
#             decorrelated_data, U1, U2 = run_decorrelate(inst_data)
            
#             self.all_decorrelated_data.append(decorrelated_data)
#             self.all_umat.append([U1,U2])
            
        return self.all_decorrelated_data, self.all_u1, self.all_u2
    
    def reverse_decorrelation(self, data, index, device = "cuda"):
        
        U1,U2 = self.all_u1[index].to(device), self.all_u2[index].to(device)
        
        return (U1@data@U2.permute(0,1,3,2))
    
        
    def load_bp(self, path, file_name = None, variable_name = "PSL", sides = ["side", "bottom", "top"]):
        if file_name is None:
            file_name = "summit.20220527.hires_atm.hifreq_write.F2010.ne120pg2_r0125_oRRS18to6v3.eam.h5.0001-01-01-00000"
            print("load default data:", file_name)
        
        all_data = []
        for side in sides:
            path_2 = "%s_%s"%(file_name, side)
            path_3 = "%s.bp.%s_mgard"%(path_2, variable_name)
            bp_path = os.path.join(path, file_name, path_2, path_3)
            
            with adios2.open(bp_path, 'r') as bp_file:
                variable = np.asarray(bp_file.read(variable_name))
                
                if side == "side":
                    variable = variable.reshape([720, 240, 960])
                else:
                    variable = variable.reshape([720, 240, 240])
                
            all_data.append(variable)
        
        self.all_data = np.concatenate(all_data,  axis = 2).reshape([720,1,240,6,240]).transpose([0,1,3,2,4])
        
        
        
    def __len__(self):
        return self.dataset_length
    
    def cutting_frame(self,idx, normalize = True):
        path_idx  = idx//(self.n_side*self.n_patch)*  self.n_frame
        side_idx  = idx% (self.n_side*self.n_patch)// self.n_patch
        patch_idx = idx% (self.n_side*self.n_patch)%  self.n_patch
        prow      = patch_idx//self.side_length * self.cut_step
        pcol      = patch_idx% self.side_length * self.cut_step
        
        

        
        if self.pre_load:
            input_data = self.all_data[path_idx:path_idx+self.n_frame, :, side_idx, prow:(prow+self.patch_size), pcol:(pcol+self.patch_size)]
            input_data = input_data.transpose([1,0,2,3])
        
        else:
            input_data = []
        
            for i in range(self.n_frame):
                data = np.fromfile(self.all_paths[path_idx+i],dtype = np.float32).reshape([-1, 6, 240, 240])
                data = data[:, side_idx, prow:(prow+self.patch_size), pcol:(pcol+self.patch_size)]   
                input_data.append(data[:,None]) # [N, 1, 16, 16]

            input_data = np.concatenate(input_data, axis=1)
        
        if normalize == True:
            input_data = (input_data - self.var_mean[0])/self.var_std[0]
        elif normalize=="min_max":
            input_data = (input_data - self.x_min[0])/(self.x_max[0] - self.x_min[0])
            
        return input_data
    
        

    def __getitem__(self, idx):
        
        input_data = self.cutting_frame(idx, (not self.norm_each) and self.normalize)
        
        if self.norm_each:
            input_data = (input_data - self.inst_means[idx])/self.inst_stds[idx]
        
        input_data = torch.from_numpy(input_data.astype(np.float32))
        
        if self.flatten:
            input_data = input_data.reshape([-1])
            
            
            if self.input_size>0:
                n_zeros = self.input_size - input_data.shape[0]
                input_data = torch.nn.functional.pad(input_data, (0, n_zeros))
                
                
            
            
        
        if self.extra_dim:
            input_data = input_data[None]
        
        output_list = [input_data]
        
        
        
        
        if self.decorrelate_frame:
            output_list.append(self.all_decorrelated_data[idx])
        
        if self.return_path:
            output_list.append(self.all_paths[path_idx])
        
        if self.return_index:
            output_list.append(idx)
            
        
        if self.norm_each:

            output_list.extend([self.inst_means[idx], self.inst_stds[idx]])
            
            
        if len(output_list)==1:
            return output_list[0]
    

        return output_list
    
    
    
# def recover_data_format(data):
#     shape = data.shape
#     patch_size = data.shape[-1]
#     n_frame = data.shape[-3]
#     n_side = 6
#     side_length = int(240//patch_size)
#     n_patch = int(side_length * side_length)
    
#     cur_frame = np.zeros([720, 6, 240, 240])
#     assert (cur_frame.size == data.size)
    
#     for idx in range(len(data)):
#         path_idx  = idx//(n_side*n_patch)*  n_frame
#         side_idx  = idx% (n_side*n_patch)// n_patch
#         patch_idx = idx% (n_side*n_patch)%  n_patch
#         prow      = patch_idx//side_length * patch_size
#         pcol      = patch_idx% side_length * patch_size
#         cur_frame[path_idx:(path_idx+n_frame),  side_idx, prow:(prow+patch_size), pcol:(pcol+patch_size)] = data[idx]
    
#     return cur_frame

def recover_data_format(data, size = 240):
    shape = data.shape
    patch_size = data.shape[-1]
    n_frame = data.shape[-3]
    n_side = 6
    side_length = int(size//patch_size)
    n_patch = int(side_length * side_length)
    
    cur_frame = np.zeros([720, 6, size, size])
    assert (cur_frame.size == data.size)
    
    for idx in range(len(data)):
        path_idx  = idx//(n_side*n_patch)*  n_frame
        side_idx  = idx% (n_side*n_patch)// n_patch
        patch_idx = idx% (n_side*n_patch)%  n_patch
        prow      = patch_idx//side_length * patch_size
        pcol      = patch_idx% side_length * patch_size
        cur_frame[path_idx:(path_idx+n_frame),  side_idx, prow:(prow+patch_size), pcol:(pcol+patch_size)] = data[idx]
    
    return cur_frame

def load_data(dpath, patch_size=16):
    data_path="/data2/xiaoli/climate_dataset/dataset_numpy"

    dataset_mgard = ClimatePatch(data_path =data_path, seqs = [8],  return_path = False,
                               mode = "TC1", patch_size = patch_size, n_frame = 20,
                               flatten=False, pre_load=True)
    file_name = "summit.20220527.hires_atm.hifreq_write.F2010.ne120pg2_r0125_oRRS18to6v3.eam.h5.0001-08-29-00000"
    dataset_mgard.load_bp(dpath, file_name, "PSL",  ["side", "bottom", "top"])
    dataset_mgard.pre_load = True
    mgard_data_reshape = []
    mgard_loader = DataLoader(dataset_mgard, batch_size=256, shuffle=False, num_workers = 4)
    for data  in (mgard_loader):
        mgard_data_reshape.append(data)
    mgard_data_reshape = np.concatenate(mgard_data_reshape, axis = 0)
    mgard_data_reshape = mgard_data_reshape * dataset_mgard.var_std + dataset_mgard.var_mean
    
    mgard_data_reshape = mgard_data_reshape.reshape([-1,1, patch_size,patch_size])
    return mgard_data_reshape
