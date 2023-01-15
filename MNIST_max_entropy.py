import os #; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

class CNN_MNIST(nn.Module):

    def __init__(self,config):
        super(CNN_MNIST, self).__init__()
        # CNN for MNIST dataset
        self.c = config
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        # Start by sleeping 
        self.wake = True
        
        # Useful for experiments 
        self.can_sleep = self.c.can_sleep
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.relu(x)
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
                    for batch_idx, (data, target) in enumerate(self.train_data):
                            self.wake_optimizer.zero_grad()
                            data = data.to(self.device)
                            target = target.to(self.device)
                            output = self.model(data)
                            loss = self.criterion(output, target)

                        
                            loss.backward()
                            self.wake_optimizer.step()
                
                            save_loss.append(loss.item())
                        
                                
                            if batch_idx % 100 == 0:
                                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(e+1, batch_idx * len(data), len(self.train_data.dataset),100. * batch_idx / len(self.train_data), loss.item()))
                        
            print('Wake epoch:', e+1)
            accuracy  = self.eval().item()
            save_accuracy.append(accuracy)
            if self.model.can_sleep == True:
                self.model.wake = False
            if self.model.wake == False:
                for i in range(self.sleep_itr):
                    noise_injection = torch.randn(64,1,28,28)
                    self.sleep_optimizer.zero_grad()
                    data = noise_injection.to(self.device)
                    output = self.model(data)
                    entropy = -torch.sum(output*torch.log(output))
                    loss = -entropy
                    loss.backward()
                    self.sleep_optimizer.step()
                    # Add los    # Save the plot to a file
    #plt.savefig('plot.png')h:', e+1)
            sleep_accuracy  = self.eval().item()
            sleep_accuracy_list.append(sleep_accuracy)
            self.model.wake = True
        # Plot loss and accuracy
        self.logger.add_log(self.c.job_num,'Wake loss', save_loss)
        self.logger.add_log(self.c.job_num,'Wake accuracy', save_accuracy)
        self.logger.add_log(self.c.job_num,'Entropy', save_entropy)
        self.logger.add_log(self.c.job_num,'Sleep accuracy', sleep_accuracy_list)
        self.logger.write_to_csv('log.csv')
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

class Logger:
    def __init__(self):
        self.run_info = {}

    def add_log(self,run_num,name,values):
        # Add the infos to the run_info dictionary
        run_num = str(run_num)
        if run_num not in self.run_info:
            self.run_info[run_num] = {}
        self.run_info[run_num][name] = values
    
    def write_to_csv(self, file_name):
        # Create DataFrame from input dictionary
        df = pd.DataFrame.from_dict(self.run_info, orient='index')
        if os.path.isfile (file_name):
            old_df = pd.read_csv(file_name)
            df = pd.concat([old_df, df], ignore_index=True)
        df.to_csv(file_name, index=False)
    
        
    def plot_with_std(self,data, labels):
        # Plot the data
        for i, d in enumerate(data):
            plt.plot(d, label=labels[i])

        # Compute the standard deviation
        std = np.std(data, axis=0)

        # Plot the standard deviation transparently behind the run lines
        plt.fill_between(range(len(std)), np.min(data, axis=0)-std, np.max(data, axis=0)+std, alpha=0.2) # alpha is the transparency
        plt.legend()
        plt.show()
        plt.savefig('plot.png')
        
            
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
    main()
    print('Finished correctly')