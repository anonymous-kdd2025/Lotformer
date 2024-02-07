import numpy as np
import pandas as pd

class Splitter:
    def __init__(self, data, split=None, eps=None, left_padding=0, right_padding=0):
        # data 1darray
        self.left = data.min() - left_padding
        self.right = data.max() + right_padding
        if split is not None and eps is not None:
            self.split = split
            self.eps = eps
        elif eps is None:
            self.eps = (self.right - self.left) / split
            self.eps = eps
            self.split = split
        # elif split is None:
        else:
            self.eps = eps
            self.split = int(np.floor((self.right - self.left) / self.eps))
        
        class_data = np.floor((data - self.left) / self.eps).astype(np.int64)
        class_data[class_data>self.split-1] = self.split - 1
        
        freq = np.bincount(class_data, minlength=self.split)
        freq[freq == 0] = 1e9
        
        self.weight = 1 / freq * self.split / (1 / freq).sum()
        # print(self.weight)
        

    def value_to_class(self, values):
        # values: torch or numpy
        c = np.floor((values - self.left) / self.eps)
        c[c>self.split-1] = self.split-1
        return c

    def class_to_value(self, c):
        return self.left + (c + 0.5) * self.eps
    
    def get_weight(self, c):
        return self.weight[c]
    
    def prob_to_exp(self, prob, greedy=True):
        # prob: batch_size, dim_class
        if greedy:
            return (prob.argmax(dim=1) + 0.5) * self.eps + self.left
        return (prob.to('cpu') * (self.left + self.eps * (torch.range(0, prob.shape[1] - 1) + 0.5))).sum(dim=1)
    
class UniformSplitter:
    def __init__(self, data, split_num, name='Y', pos_noise=None):
        # data: pandas
        n = len(data)
        self.pos_noise = pos_noise
        if pos_noise is not None:
            data.loc[:, name] += np.random.normal(0, pos_noise, size = n)
        self.split_point = data.sort_values(by=name).reset_index(drop=True).loc[[ int(i / split_num * n) for i in range(1, split_num)], name].values
        self.left, self.right = data.min(), data.max()
        middle = np.concatenate([np.asarray(self.left), self.split_point, np.asarray(self.right)])
        self.middle = (middle[1:] + middle[:-1]) / 2 # weighted average is a possible choice
        # print(self.middle)
        left_middle = data[data[name] <= middle[1]].mean().values
        right_middle = data[data[name] >= middle[-2]].mean().values
        # print(left_middle, right_middle)
        self.middle[0] = left_middle
        self.middle[-1] = right_middle
        self.weight = None
        
    def value_to_class(self, values):
        if self.pos_noise is not None:
            values = values.copy().astype(np.float64)
            values += np.random.normal(0, self.pos_noise, size=values.shape)

        if len(values.shape) == 1:
            values = np.expand_dims(values, axis=1)
        return (values > self.split_point).sum(axis=1)
    
    def class_to_value(self, c):
        return self.middle[c]
    
    def prob_to_exp(self, prob, greedy=False):
        if greedy:
            return torch.FloatTensor(self.middle)[(prob.argmax(dim=1).to('cpu').detach().numpy())]
        return (prob.to('cpu').detach() * torch.FloatTensor(self.middle)).sum(axis=1)
        
        

def get_mapping_data(min_data, max_data, eps, class_num):
    assert min_data < max_data
    return min_data - eps, (max_data - min_data + 2 * eps) / class_num