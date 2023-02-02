import pandas as pd
import cv2 
from torch.utils.data import IterableDataset, DataLoader
import random
import torch 
import torch.nn as nn
import numpy as np

class EntropyHook:
    def __init__(self, modules):
        self.entropy_sum = 0
        self.num_layers = 0
        self.hooks = [module.register_forward_hook(self.hook_fn) for module in modules if isinstance(module, nn.Linear)]

    def hook_fn(self, module, input, output):
        # Compute softmax of the output
        softmax_output = torch.softmax(output, dim=-1)
        # Compute entropy of the softmax output
        entropy = -(softmax_output * torch.log(softmax_output)).sum(dim=-1).mean()
        # Accumulate the entropy
        self.entropy_sum += entropy.item()
        self.num_layers += 1

    def remove(self):
        for hook in self.hooks:
            hook.remove()

    def get_entropy_sum(self):
        return self.entropy_sum 
    
    def clear(self):
        self.entropy_sum = 0
        self.num_layers = 0


class Logger:
    def __init__(self):
        self.log = {}
    def add_log(self,feature_name,value):
        self.log[feature_name] = value       
    def fill_missing_values(self,data: dict) -> dict:
        max_len = max([len(v) for v in data.values()])
        for key in data:
            if len(data[key]) < max_len:
                data[key] += [None] * (max_len - len(data[key]))
        return data
    def write_to_csv(self, file_name):
        filled_data = self.fill_missing_values(self.log)
        df = pd.DataFrame(filled_data)
        df.to_csv(file_name, index=False)
        
    def write_video(self,filepath,frames, fps=60):
        """ Write a video to disk using openCV
            filepath : the path to write the video to 
            frames : a numpy array with shape (time, height, width, channels)
            
        
        """
        height, width, channels = frames.shape[1:]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()

def set_seed(seed : int,device):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed_all(seed)