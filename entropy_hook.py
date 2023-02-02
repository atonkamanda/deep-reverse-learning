import torch
import torch.nn as nn

class EntropyHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        softmax_output = torch.softmax(output, dim=-1)
        
        entropy = -torch.sum(softmax_output*torch.log(softmax_output))
        #entropy = -(softmax_output * torch.log(softmax_output)).sum(dim=-1).mean()
        setattr(output, 'entropy', entropy)

    def remove(self):
        self.hook.remove()

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.fc = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.fc2(x)
        return x

# Use the hook
model = MyModule()
entropy_hooks = [EntropyHook(layer) for layer in model.modules()]
#input = torch.randn(1, 10)
# Create random uniform flat input
input = torch.ones(1, 10) / 10
#print(input)
output = model(input)
print(output.entropy)
# Access the entropy from the output tensor
EntropyHook(model)
output = model(input)
print(output.entropy)


