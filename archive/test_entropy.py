
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pathlib 
from torchvision.models.feature_extraction import get_graph_node_names,create_feature_extractor
import matplotlib.pyplot as plt


"""# Create a tensor of 10 logits uniformly distributed between 0 and 1
#logits = torch.rand(10, requires_grad=False)
# Create a tensor of logits with a low uncertainty
probs = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], requires_grad=False)
#probs = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.91], requires_grad=False) 
# Create a tensor of shape (1000,) with the same value
n_categories = 10
#logits = torch.full((n_categories,), 0.1, requires_grad=False)
#logits = torch.randn(n_categories, requires_grad=False)
# Apply a softmax to the logits
#probs = F.softmax(logits, dim=0)


# Plot the probs as a bar chart with x axis labels and y axis softmax values
plt.bar(range(n_categories), probs, color='green')
plt.xticks(range(n_categories), range(n_categories))
plt.xlabel('Class')
plt.ylabel('Softmax')
# Add a digit in the plot in the legend to show the entropy of the softmax 
entropy = -torch.sum(probs*torch.log2(probs), dim=0)
plt.legend(['Entropy: {:.2f}'.format(entropy)])
# Change color of the legend
plt.show()
# Save plot
plt.savefig('entropy.png')"""


unif = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
probs = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.91]
# Create a single tensor with the two softmaxes
x = torch.tensor([unif, probs], requires_grad=False)

# For each element of the batch compute the entropy 
entropy = -torch.sum(x*torch.log2(x), dim=1)
print(entropy)