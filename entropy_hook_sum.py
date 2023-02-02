import torch
import torch.nn as nn

import torch

class EntropyHook:
    def __init__(self, module):
        self.module = module
        self.entropy_sum = 0
        
    def __call__(self, module, input, output):
        self.module = module
        self.input = input
        logsoftmax_output = torch.nn.functional.log_softmax(output, dim=1)
        entropy = -(logsoftmax_output * torch.nn.functional.softmax(output, dim=1)).sum(dim=1)
        self.entropy_sum += entropy.sum().item()

def add_entropy_hooks(model):
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Module):
            hooks.append(module.register_forward_hook(EntropyHook(module)))
    return hooks

def remove_entropy_hooks(hooks):
    for hook in hooks:
        hook.remove()


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
input = torch.randn(64, 10)
# Example usage:
model = MyModule()
hooks = add_entropy_hooks(model)
output = model(input)

remove_entropy_hooks(hooks)
print("Sum of entropy:", hooks[-1].entropy_sum)




# Example usage
model = MyModule()
hook = EntropyHook(model.fc1)
output = model(input)
entropy = hook.output
hook.remove()
# Use the hook
model = MyModule()
entropy_hook = EntropyHook(model.modules())

model(input)

# Get the sum of the entropies
print(f"The average of the entropies of all the layers is: {entropy_hook.get_entropy_sum()}")

# Remove the hooks
entropy_hook.remove()



# Use the hook
model = MyModule()
entropy_hook = EntropyHook(model.modules())
input = torch.randn(64,10)
model(input)

# Get the sum of the entropies
print(f"The sum of the entropies of all the layers is: {entropy_hook.get_entropy_sum()}")

# Remove the hooks
entropy_hook.remove()
