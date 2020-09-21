import torch
import torch.nn.functional as  F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

EPOCH=30
if torch.cuda.is_available():
    device=torch.device("cuda")
    print("采用了GPU加速")
else:
    device=torch.device("cpu")
    print("GPU无法正常使用")
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(28 * 28, 300)
        self.hidden2 = torch.nn.Linear(300, 100)
        self.output = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        return x

net = Net()
net=net.to(device)

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_set = datasets.MNIST('../pytorch_', train=True, download=True, transform=transform)
test_set = datasets.MNIST('../pytorch_', train=False, download=True, transform=transform)

train_data = DataLoader(train_set, batch_size=128, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=True)

for t in range(EPOCH):

    train_correct=0
    loss_all=0
    acc_all=0
    net.train()
    for i,(x,y) in enumerate(train_data):
        x,y=x.to(device),y.to(device)
        x=x.view(-1,28*28)
        out=net(x)
        loss=loss_func(out,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _,pre=out.max(1)
        train_correct=(pre==y).sum().float()
        acc=train_correct/x.shape[0]
        loss_all+=float(loss)
        acc_all+=acc

    print('epoch:{},acc:{:.4f},loss:{:.4f}'.format(t+1,acc_all/len(train_data),loss_all))
    acc_test=0
    loss_test=0
    net.eval()
    for i,(x,y) in enumerate(test_data):
        x,y=x.to(device),y.to(device)
        x=x.view(-1,28*28)
        test_out=net(x)
        test_loss=loss_func(test_out,y)
        _,pre=test_out.max(1)
        test_correct=(pre==y).sum().float()
        test_acc=train_correct/x.shape[0]
        loss_test+=float(test_loss)
        acc_test+=test_acc

    print('epoch:{},test_acc:{:.4f},test_loss:{:.4f}'.format(t+1,acc_test/len(test_data),loss_test))


