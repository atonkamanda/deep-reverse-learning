import torch

# Create a tensor of values
values = torch.tensor([0.25, 0.25, 0.25, 0.25])
#values = torch.tensor([1.0, 2.0, 3.0, 4])

# Use the Softmax function to transform the values into a probability distribution
distribution = torch.softmax(values, dim=0)

# Print the resulting distribution
print(distribution)

# Compute and print the entropy of the distribution
entropy = -(distribution * torch.log(distribution)).sum()
print("Entropy:", entropy)
