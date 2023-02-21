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
from utils import Logger,EntropyHook
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
    sleep_itr : int = 10000
    wake_lr : float = 0.02
    sleep_lr : float = 0.001    
    
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

class MLP(nn.Module):

    def __init__(self, config:Config):
        super(MLP, self).__init__()
        self.c = config
        # Model
        self.l1 = nn.Linear(784, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)
        
        # Start by sleeping 
        self.wake = True
        
        # Useful for experiments 
        self.can_sleep = self.c.can_sleep
        # Add dropout
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        #(n, 1, 28, 28)-> (n, 784)
        """x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
  
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.softmax(self.l5(x), dim=1)"""
        entropy = 0 
        x = x.view(-1, 784)
        x = self.l1(x)
        # Compute e
        entropy += torch.sum(-F.softmax(x, dim=1) * F.log_softmax(x, dim=1), dim=1)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        # Compute e
        entropy += torch.sum(-F.softmax(x, dim=1) * F.log_softmax(x, dim=1), dim=1)
  
        x = self.l3(x)
        x = F.relu(x)
        # Compute e
        entropy += torch.sum(-F.softmax(x, dim=1) * F.log_softmax(x, dim=1), dim=1)
        x = self.l4(x)
        x = F.relu(x)
        # Compute e
        entropy += torch.sum(-F.softmax(x, dim=1) * F.log_softmax(x, dim=1), dim=1)
        x = self.l5(x)
        # Compute e
        entropy += torch.sum(-F.softmax(x, dim=1) * F.log_softmax(x, dim=1), dim=1)
        x = F.softmax(x, dim=1)
        
        return x,entropy

        

class Trainer:
    def __init__(self, config:Config):
        self.c = config
        self.seed = self.c.seed
        self.device = self.c.device
        self.logger = Logger()
        self.train_data, self.test_data = DataManager(self.c).load_MNIST()
        self.model = MLP(self.c).to(self.c.device)
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
                            
                            output,_ = self.model(data)
                            loss = self.criterion(output, target)

                        
                            loss.backward()
                            self.wake_optimizer.step()
                
                            save_loss.append(loss.item())
                        
                                
                            if batch_idx % 100 == 0:
                                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(e+1, batch_idx * len(data), len(self.train_data.dataset),100. * batch_idx / len(self.train_data), loss.item()))
                        
                    
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
                    
                    output,entropy = self.model(data)
                 
                    entropy= -torch.sum(output*torch.log2(output),dim=1)
                    loss = -entropy
                    loss.backward()
                    self.sleep_optimizer.step()
            #print(f"The sum of the entropies of all the layers is: {self.entropy_hook.get_entropy_sum()}")       
                sleep_accuracy  = self.eval().item()
                sleep_accuracy_list.append(sleep_accuracy)
                self.model.wake = True
                self.entropy_hook.remove()
        # Plot loss and accuracy
        self.logger.add_log('Wake loss', save_loss)
        self.logger.add_log('Wake accuracy', save_accuracy)
        self.logger.add_log('Entropy', save_entropy)
        self.logger.add_log('Sleep accuracy', sleep_accuracy_list)
        self.logger.write_to_csv('log.csv')
    def eval(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.test_data:
            data = data.to(self.device)
            target = target.to(self.device)
            output,_ = self.model(data)
        
            # sum up batch loss
            test_loss += torch.mean(self.criterion(output, target)).item()
            # Compute accuracy 
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        test_loss /= len(self.test_data.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(self.test_data.dataset),100. * correct / len(self.test_data.dataset)))
        # Return accuracy 
        return correct / len(self.test_data.dataset)

        
            
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    # Load default config
    default_config = OmegaConf.structured(Config)
    # Merge default config with run config, run config overrides if there is a conflict
    config = OmegaConf.merge(default_config, cfg)
    #OmegaConf.save(config, 'config.yaml') 
    
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    job_num = hydra_cfg.job.num
    print(f'Hydra job number: {job_num}')
    config.job_num = job_num
    
    trainer = Trainer(config)
    trainer.train(config.epoch)
    
    
if __name__ == '__main__':
    print("TEst")
    main()
    print('Finished correctly')