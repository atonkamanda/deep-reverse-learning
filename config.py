import torch

class Config:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 64
        self.num_workers = 4
        self.epochs = 10
        self.momentum = 0.9
        self.log_interval = 100
        self.save_model = True
        self.model_path = './model.pth'
        self.model = None

        # reproducibility parameters
        self.seed = 42
        self.load_model = False

        # logging parameters
        self.log_dir = './logs'
        self.log_every = 100
        self.print_every = 10

        # wake sleep parameters

        # task parameters

        ## data parameters
        self.dataset = 'CIFAR10' #'MINIST' # 'CIFAR10', 'CIFAR100'
        self.mixup = False  
        self.alpha = 4

        ## training
        self.epoch = 10      # nbr of epoch

        # model parameters
        self.batch_size = 64
        self.can_sleep = False  # if True, the model can sleep

        self.wake_itr = 3000     # nbr of iteration in the wake phase
        self.sleep_itr = 500    # nbr of iteration in the sleep phase

        self.wake_lr = 0.01    # learning rate in the wake phase
        self.momentum = 0.5   # momentum in the wake phase
        self.sleep_lr = 0.2   # learning rate in the sleep phase