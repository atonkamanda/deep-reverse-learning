import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pathlib
import matplotlib.pyplot as plt
from dataclasses import dataclass
from omegaconf import OmegaConf,DictConfig
import hydra
import pickle
import numpy as np
import pandas as pd
from utils import Logger
from termcolor import colored
import time  
@dataclass
class Config:
    
    # Reproductibility and hardware 
    seed : int = 0
    device : str = 'cuda' if torch.cuda.is_available() else 'cpu'
    load_model : bool = False
    job_num : int = 0

    # Logging
    logdir : str = pathlib.Path.cwd() / 'logs'
    log_every : int = 1000
    print_every : int = 10
    
    
    
    # Task hyperparameters 
    dataset : str = 'MNIST'
    epoch : int = 10 # The number of update 
    
    
    # Model hyperparameters
    batch_size : int = 64
    can_sleep : bool = True
    sleep_itr : int = 1000
    wake_lr : float = 0.02
    sleep_lr : float = 0.01    

    

    
class DataManager():
    def __init__(self,config):
        self.c = config
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        
    def load_MNIST(self):
        # Loading MNIST
        train_dataset = datasets.MNIST(root='./data/',
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True,
                                    )

        test_dataset = datasets.MNIST(root='./data/',
                                    train=False,
                                    transform=transforms.ToTensor())


        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=self.c.batch_size,
                                                shuffle=True,
                                                pin_memory=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=self.c.batch_size,
                                                shuffle=False,
                                                pin_memory=True)
        return train_loader,test_loader

# Initialize MLP

class CNN_MNIST(nn.Module):

    def __init__(self,config):
        super(CNN_MNIST, self).__init__()
        # CNN for MNIST dataset
        self.c = config
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.wake = True
        
        # Useful for experiments 
        self.can_sleep = self.c.can_sleep
        
        # Add dropout 
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 320)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        # Compute entropy of the output distribution
            
        
        return x 
  

class Trainer:
    def __init__(self, config:Config):
        self.c = config
        self.seed = self.c.seed
        self.device = self.c.device
        self.logger = Logger()
        self.train_data, self.test_data = DataManager(self.c).load_MNIST()
        self.model = CNN_MNIST(self.c).to(self.c.device)
        self.sleep_itr = self.c.sleep_itr
        self.criterion = nn.CrossEntropyLoss()
        self.wake_optimizer = optim.SGD(self.model.parameters(), lr=self.c.wake_lr, momentum=0.5)
        self.sleep_optimizer = optim.SGD(self.model.parameters(), lr=self.c.sleep_lr, momentum=0.5)
    def train(self, epoch):
        self.model.train()
        save_loss = []
        save_accuracy = []
        save_entropy = []
        sleep_accuracy_list = []
        
        for e in range(epoch):
            if self.model.wake == True:
                print('Wake epoch:', e+1)
                for batch_idx, (data, target) in enumerate(self.train_data):
                    self.wake_optimizer.zero_grad()
                    data = data.to(self.device)
                    target = target.to(self.device)
                    output = self.model(data)
                    loss = self.criterion(output, target)

                
                    loss.backward()
                    
                    
                    self.wake_optimizer.step()
                    # Add gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

        
                    save_loss.append(loss.item())
                
                        
                    if batch_idx % 100 == 0:
                        epoch = 'E: {:.0f} '.format(e+1, epoch)
                        loss_line = 'Loss: {:.6f} '.format(save_loss[-1])
                        percent = '(Completion: {:.0f}%) '.format(100. * batch_idx / len(self.train_data))
                        
                        
                        print(colored(epoch, 'cyan'), end='')
                        print(colored(loss_line, 'red'), end=' ') 
                        print(colored(percent, 'yellow'), end='\n')

        
            accuracy  = self.eval().item()
            save_accuracy.append(accuracy)  
            
            
            if self.model.can_sleep == True:
                self.model.wake = False
            if self.model.wake == False:
                print('Sleep epoch:', e+1)
                for i in range(self.sleep_itr):
                    noise_injection = torch.rand(64,1,28,28)
                    self.sleep_optimizer.zero_grad()
                    data = noise_injection.to(self.device)
                    output = self.model(data)
                    entropy = -torch.sum(output*torch.log2(output), dim=1)
                    entropy = torch.sum(entropy)
                    loss = -entropy
                    loss.backward()
                    # Add gradient clipping
                    #torch.nn.utils.clip_grad_norm_(self.model.parameters(),5)
                    self.sleep_optimizer.step()
             
                    if i % 100 == 0:
                        epoch = 'E: {:.0f} '.format(e+1, epoch)
                        loss_line = 'Entropy: {:.6f} '.format(loss)
                        percent = '(Completion: {:.0f}%) '.format(100. * i / self.sleep_itr)
                
                            

                        print(colored(epoch, 'cyan'), end='')
                        print(colored(loss_line, 'red'), end=' ')
                        print(colored(percent, 'magenta'), end='\n')
                print('Sleep accuracy at epoch', e+1)
                sleep_accuracy  = self.eval().item()
                sleep_accuracy_list.append(sleep_accuracy)
                self.model.wake = True
    
            
    def eval(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.test_data:
            data = data.to(self.device)
            target = target.to(self.device)
            output = self.model(data)
        
            # sum up batch loss
            test_loss += torch.mean(self.criterion(output, target)).item()
            # Compute accuracy 
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        test_loss /= len(self.test_data.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(self.test_data.dataset),100. * correct / len(self.test_data.dataset)))
        # Return accuracy 
        return correct / len(self.test_data.dataset)

            
#@hydra.main(version_base=None, config_path="conf", config_name="config")
def main() -> None: # cfg : DictConfig
    # Load default config
    default_config = OmegaConf.structured(Config)
    # Merge default config with run config, run config overrides if there is a conflict
    #config = OmegaConf.merge(default_config, cfg)
    #OmegaConf.save(config, 'config.yaml') 
    config = default_config
    
    #hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    """job_num = hydra_cfg.job.num
    print(f'Hydra job number: {job_num}')
    config.job_num = job_num"""
    
    trainer = Trainer(config)
    trainer.train(config.epoch)
    
    
if __name__ == '__main__':
    main()
    print('Finished correctly')