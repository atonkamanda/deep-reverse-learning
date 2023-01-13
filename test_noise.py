import torch
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

noise_injection = torch.randn(2,28,28)


def display_image(image,name):
  plt.imshow(image, cmap='viridis')
  plt.show()
  # Save the image
  plt.imsave(name+'.png',image, cmap='viridis')
  


# Display the noise
#display_image(noise_injection[0],'noise_1')
#display_image(noise_injection[1],'noise_2')
# Print the noise
print(noise_injection[0][0])
print("dif")
print(noise_injection[1][0])