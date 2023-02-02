

# Create an uniform distribution of 10 classes and compute the entropy
import torch
import torch.nn as nn
import torch.nn.functional as F
random = torch.rand(1, 10)
# Compute the entropy
entropy = -torch.sum(random*torch.log(random))
print(entropy)