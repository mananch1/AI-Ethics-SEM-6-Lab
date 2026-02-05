import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
import os

os.makedirs("data", exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor()
])

# Load Fashion-MNIST
train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Split training data into 3 clients
length = len(train_dataset) // 3
client1, client2, client3 = random_split(
    train_dataset,
    [length, length, len(train_dataset) - 2 * length]
)

torch.save(client1, "data/client1.pt")
torch.save(client2, "data/client2.pt")
torch.save(client3, "data/client3.pt")
torch.save(test_dataset, "data/test.pt")

print("Fashion-MNIST successfully split into 3 clients + server test set.")
