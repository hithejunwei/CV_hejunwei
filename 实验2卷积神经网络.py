import torch
import torch.nn.functional as  F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

EPOCH=20
if torch.cuda.is_available():
    device=torch.device("cuda")
    print("采用了GPU加速")
else:
    device=torch.device("cpu")
    print("GPU无法正常使用")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = datasets.CIFAR10('../pytorch_', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10('../pytorch_', train=False, download=True, transform=transform)

train_data = DataLoader(train_set, batch_size=128, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=True)
net=torch.nn.Sequential(

)

