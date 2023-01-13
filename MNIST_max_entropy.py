import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pathlib 
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.feature_extraction import get_graph_node_names,create_feature_extractor
import matplotlib.pyplot as plt


# Hyperparams   
batch_size = 60
# set device variable to cuda if cuda is available if not set to cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set the seed
torch.manual_seed(30)

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
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          pin_memory=True)

log_dir = pathlib.Path.cwd() / 'logs'
writer = SummaryWriter(log_dir=log_dir)


# Initialize MLP

class CNN_MNIST(nn.Module):

    def __init__(self):
        super(CNN_MNIST, self).__init__()
        # CNN for MNIST dataset
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        # Start by sleeping 
        self.wake = True
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

        



def train(epoch,model):
    model.train()
    save_loss = []
    save_accuracy = []
    save_entropy = []
    sleep_accuracy_list = []
    for e in range(epoch):
        if model.wake == True:
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    data = data.to(device)
                    target = target.to(device)
                    output = model(data)
                    loss = criterion(output, target)

                    
                    loss.backward()
                    optimizer.step()
                    # Add loss to the list
                    save_loss.append(loss.item())
                    
                            
                    if batch_idx % 100 == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(e+1, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))
                
                print('Wake epoch:', e+1)
                accuracy  = test().item()
                save_accuracy.append(accuracy)
                #odel.wake = False
        if model.wake == False:
            for i in range(10000):
                noise_injection = torch.randn(60,1,28,28)
                optimizer2.zero_grad()
                data = noise_injection.to(device)
                output = model(data)
                entropy = -torch.sum(output*torch.log(output))
                loss = -entropy
                loss.backward()
                optimizer2.step()
                # Add loss to the list
                save_entropy.append(entropy.item())
                if i % 5000 == 0:
                    print(entropy.item())
                
                    
                    
            
            print('Sleep Epoch:', e+1)
            sleep_accuracy  = test().item()
            sleep_accuracy_list.append(sleep_accuracy)
            
            model.wake = True
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
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
    
        # sum up batch loss
        test_loss += torch.mean(criterion(output, target)).item()
        # Compute accuracy 
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    # Return accuracy 
    return correct / len(test_loader.dataset)
 



model = CNN_MNIST().to(device)
criterion = nn.CrossEntropyLoss()
criterion_critic = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.5)
optimizer2 = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
"""optimizer = optim.SGD([
                {'params': model.l1.parameters(),'lr': 2e-1},
                {'params': model.l2.parameters(),'lr': 2e-1},
                {'params': model.l3.parameters(),'lr': 2e-1},
                {'params': model.l4.parameters(),'lr': 2e-1},
                {'params': model.l5.parameters(),'lr': 2e-1}],momentum=0.5)"""
train(10,model)