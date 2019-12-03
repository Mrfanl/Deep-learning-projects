import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

EPOCH = 5
BATCH_SIZE = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

trainset = datasets.MNIST('E:\\dl-dataset', train=True, download=False, transform=transform)
testset = datasets.MNIST('E:\\dl-dataset', train=False, download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)


class SoftmaxNet(nn.Module):
    def __init__(self):
        super(SoftmaxNet, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        a = self.fc1(x)
        a = self.fc2(a)
        return F.log_softmax(a)


net = SoftmaxNet()
optimizer = optim.SGD(net.parameters(), lr=0.05)

for epoch in range(EPOCH):
    all_loss = 0
    for idx, inputs in enumerate(trainloader, 0):
        optimizer.zero_grad()
        image, label = inputs
        image = image.view(-1, 28 * 28)
        out = net(image)
        loss = F.nll_loss(out, label)
        loss.backward()
        optimizer.step()
        all_loss += loss.item()
        if idx % 100 == 99:
            print("epoch %d, batch %d, loss:%.6f" % (epoch + 1, idx + 1, all_loss / 100.))
            all_loss = 0

torch.save(net.state_dict(), './net.pth')

net.load_state_dict(torch.load('./net.pth'))
with torch.no_grad():
    total = 0
    for idx, inputs in enumerate(testloader, 0):
        image, label = inputs
        image = image.view(-1, 28 * 28)
        out = net(image)
        total += torch.sum(label == torch.argmax(out, dim=1))
print("acc:%.4f" % (total.numpy() / testset.data.shape[0],))
