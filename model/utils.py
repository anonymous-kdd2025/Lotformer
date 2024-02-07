from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

class CTDataset(Dataset):
    # output: batch, t0, dim_output
    def __init__(self, cov, proxy, treatment, result, exg, t0, length=None):
        self.x = torch.from_numpy(cov.values.astype(np.float32)).unsqueeze(1).repeat(1, t0, 1)
        self.m = torch.from_numpy(proxy.values.astype(np.float32)).view(-1, t0, proxy.shape[1] // t0)
        self.t = torch.from_numpy(treatment.values.astype(np.float32)).unsqueeze(-1).repeat(1, t0, 1)
        self.y_c = torch.from_numpy(exg.value_to_class(result.values)).long()
        self.y = torch.from_numpy(result.values.astype(np.float32))
        # self.data = torch.from_numpy(pd.concat([cov, panel], axis=1).values.astype(np.float32))
        self.length = length
    
    def __len__(self):
        return len(self.x) if self.length is None else self.length
    
    def __getitem__(self, idx):
        return self.x[idx], self.m[idx], self.t[idx], self.y_c[idx], self.y[idx]
    
    
class RDataset(Dataset):
    # output: batch, t0, dim_output
    def __init__(self, cov, proxy, treatment, result, t0):
        self.x = torch.from_numpy(cov.values.astype(np.float32)).unsqueeze(1).repeat(1, t0, 1)
        self.m = torch.from_numpy(proxy.values.astype(np.float32)).view(-1, t0, proxy.shape[1] // t0)
        self.t = torch.from_numpy(treatment.values.astype(np.float32)).unsqueeze(-1).repeat(1, t0, 1)
        self.y = torch.from_numpy(result.values.astype(np.float32)) #.unsqueeze(1)
        # self.data = torch.from_numpy(pd.concat([cov, panel], axis=1).values.astype(np.float32))
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.m[idx], self.t[idx], self.y[idx]