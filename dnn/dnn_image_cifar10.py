import matplotlib
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Option 1: Use an interactive backend if possible
try:
    matplotlib.use('TkAgg')
except ImportError:
    pass  # If TkAgg is not available, continue with default

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# Define a function to display an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('plot.png')
    plt.show()

# Get a batch of random training images with a progress bar
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Display the images
imshow(torchvision.utils.make_grid(images))

# Print the labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# Processing batches with a progress bar
for i, (images, labels) in enumerate(tqdm(trainloader, desc="Processing batches")):
    # Here you would have your training code
    pass
