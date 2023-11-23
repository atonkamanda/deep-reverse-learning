class Config:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 64
        self.num_workers = 4
        self.epochs = 10
        self.lr = 0.001
        self.momentum = 0.9
        self.log_interval = 100
        self.save_model = True
        self.model_path = './model.pth'
        self.model = None