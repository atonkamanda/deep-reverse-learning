import torch
from torch.optim import SGD
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import build_model
from config import Config

def load_data(config):
    train_dataset = datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True)

    # Create a DataLoader for the training dataset
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

def train(config, train_loader, model, optimizer):

    criterion = torch.nn.NLLLoss()

    for epoch in range(config.epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):

            optimizer.zero_grad()
            preds = model(data)

            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

def test(config, test_loader, model):
    
    for batch_idx, (data, targets) in enumerate(test_loader):
        preds = model(data)
        print(preds)
        break


if __name__ == "__main__":

    # create config and download data
    config = Config()
    train_loader, test_loader = load_data(config)

    # build model and its optimizer
    model = build_model()
    optimizer = SGD(model.parameters(), lr=config.lr, momentum=config.momentum)

    # train and test
    train_results = train(config, train_loader, model, optimizer)
    test_restults = test(config, test_loader, model)

