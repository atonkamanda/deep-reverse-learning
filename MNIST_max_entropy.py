import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pathlib 
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import argparse 
from dataclasses import dataclass
from omegaconf import OmegaConf

@dataclass
class Config:
    
    # Reproductibility and hardware 
    seed : int = 0
    device : str = 'cuda' if torch.cuda.is_available() else 'cpu'
    load_model : bool = False

    # Logging
    logdir : str = pathlib.Path.cwd() / 'logs'
    log_every : int = 1000
    print_every : int = 10
    
    
    
    # Task hyperparameters 
    dataset : str = 'MNIST'
    n_epoch : int = 11 # The number of update 
    
    
    # Model hyperparameters
    batch_size : int = 64
    can_sleep : bool = True
    sleep_itr : int = 1000
    wake_lr : float = 1e-2
    sleep_lr : float = 1e-2
    
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
        self.train_data, self.test_data = DataManager(self.c).load_MNIST()
        self.model = CNN_MNIST(self.c).to(self.c.device)
        self.sleep_itr = self.c.sleep_itr
        self.criterion = nn.CrossEntropyLoss()
        self.wake_optimizer = optim.SGD(self.model.parameters(), lr=0.02, momentum=0.5)
        self.sleep_optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.5)

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
                    # Add loss to the list
                    save_entropy.append(entropy.item())
                    if i % 5000 == 0:
                        print(entropy.item())
                    
                        
                        
                
            print('Sleep Epoch:', e+1)
            sleep_accuracy  = self.eval().item()
            sleep_accuracy_list.append(sleep_accuracy)
            self.model.wake = True
        # Plot loss and accuracy
        plt.plot(save_loss)
        plt.savefig('loss.png')
        plt.show()
        plt.plot(save_accuracy)
        plt.savefig('accuracy.png')
        plt.show()
        # Plot entropy
        plt.plot(save_entropy)
        plt.savefig('entropy.png')
        plt.show()
        # Plot sleep accuracy
        plt.plot(sleep_accuracy_list)
        plt.savefig('sleep_accuracy.png')
        plt.show()
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
            # Compute accuracy         print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(self.test_data.dataset),100. * correct / len(self.test_data.dataset)))
        # Return accuracy 
        return correct / len(test_loader.dataset)
 
        return correct / len(self.test_data.dataset)
 

def main():
    # Create config
    config = OmegaConf.structured(Config)
    command_line = OmegaConf.from_cli()
    config = OmegaConf.merge(config, command_line)
    OmegaConf.save(config, 'config.yaml')
    #OmegaConf.load('config.yaml')
    
    trainer = Trainer(config)
    trainer.train(10)
    #trainer.eval()


if __name__ == '__main__':
    main()
    print('Finished correctly')