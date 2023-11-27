from torch.optim import SGD
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader


from model import build_model
from config import Config

from termcolor import colored

import torch

def load_data(config):

    if config.dataset == 'CIFAR10':
        dataset = datasets.CIFAR10
    elif config.dataset == 'MNIST':
        dataset = datasets.MNIST

    train_dataset = dataset(root='./data', train=True, transform=transforms.ToTensor(), download=True) 
    test_dataset = dataset(root='./data', train=False, transform=transforms.ToTensor(), download=True)

    # Create a DataLoader for the training dataset
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader

def train(config, train_loader, test_loader, model, optimizer):

    criterion = torch.nn.CrossEntropyLoss()
    step = 0
    wake_count = 1
    sleep_count = 1

    cutmix = v2.CutMix(num_classes=10)
    mixup = v2.MixUp(num_classes=10)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])


    for epoch in range(config.epochs):

        wake_loss = 0
        sleep_loss = 0

        model.train()

        for itr, (data, targets) in enumerate(train_loader):

            step += 1

            # process batch
            data, targets = data.to(config.device), targets.to(config.device)

            # phase transition
            if ((step % config.sleep_itr) == 0) and not model.wake:
                model.wake = True

            elif ((step % config.wake_itr) == 0) and model.wake:
                if config.can_sleep:
                    model.wake = False
            # forward pass
            optimizer.zero_grad()

            # wake training
            if model.wake:

                outputs = model(data)[0]
                loss = criterion(outputs, targets)
                wake_loss += loss.item()
                wake_count += 1

            # sleep training
            else:

                # do mixup
                if config.mixup:
                    #data, targets, _ = mixup_data(data, targets, config.alpha, config.device)
                    data, targets = cutmix_or_mixup(data, targets)
                # make batch of pure noise
                else:
                    data = torch.rand_like(data)

                outputs, ent1, ent2, ent3 = model(data)

                loss = (outputs * outputs.log()).sum(dim=-1).mean() + criterion(outputs, targets)
                loss = loss + (ent1 * ent1.log()).sum(dim=-1).mean()
                loss = loss + (ent2 * ent2.log()).sum(dim=-1).mean()
                loss = loss + (ent3 * ent3.log()).sum(dim=-1).mean()
                loss = loss + (ent4 * ent4.log()).sum(dim=-1).mean()
                loss = loss + (ent5 * ent5.log()).sum(dim=-1).mean()
                loss /= 6

                sleep_loss += loss.item()
                sleep_count += 1

            loss.backward()
            optimizer.step()

        wake_loss /= wake_count
        sleep_loss /= sleep_count
        print(colored('Epoch: {}/{} \tWake loss: {:.6f} \tSleep Avg Entropy: {:.6f}'.format(epoch,config.epochs, wake_loss, -sleep_loss), 'yellow'), end='\n')
        test(config, test_loader, model)

def test(config, test_loader, model):
    
    model.eval()
    test_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()

    accuracy_per_class = torch.zeros(10)
    class_count = torch.zeros(10)

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            
            data = data.to(config.device)
            targets = targets.to(config.device)
            outputs = model(data)[0]

            test_loss += criterion(outputs, targets).item()
            preds = outputs.argmax(dim=1, keepdim=True)
            correct += preds.eq(targets.view_as(preds)).sum().item()

            # keep track of the normalization constant
            class_count += torch.bincount(targets, minlength=10)
            accuracy_per_class += torch.bincount(targets[preds.eq(targets.view_as(preds)).squeeze()], minlength=10)

    # compute test loss and accuracy
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    accuracy_per_class = 100. * accuracy_per_class / class_count

    print(colored('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset), accuracy), 'magenta', attrs=["bold"]), end='\n')
    
    print(colored('Accuracy per class:', 'magenta', attrs=["bold"]), end='')
    for i in range(10):
        print(colored('\t{}: {:.2f}%'.format(i, accuracy_per_class[i]), 'magenta', attrs=["bold"]), end='')
    print('\n')
    return test_loss, accuracy

if __name__ == "__main__":

    # create config and download data
    config = Config()

    # initialize random seed
    torch.manual_seed(config.seed)

    train_loader, test_loader = load_data(config)

    # build model and its optimizer
    model = build_model(config)

    optimizer = SGD(model.parameters(), lr=config.wake_lr, momentum=config.momentum)
    # train and test
    train_results = train(config, train_loader, test_loader, model, optimizer)
    test_restults = test(config, test_loader, model)

