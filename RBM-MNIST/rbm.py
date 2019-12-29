import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

class RBM(nn.Module):
    ## n_v为可见层的维度，n_h为隐藏层的维度，k为CD-k中采样轮次的K
    def __init__(self,n_v=784,n_h=100,k=1,device=torch.device("cpu")):
        super(RBM,self).__init__()
        self.W = nn.Parameter(torch.randn(n_h,n_v))
        self.v_bias = nn.Parameter(torch.zeros(n_v))
        self.h_bias = nn.Parameter(torch.zeros(n_h))
        self.k = k
        self.device = device
    
    def sample(self,p):
        return F.relu(torch.sign(p-torch.rand(p.size()).to(self.device)))
    
    def v2h(self,v):
        p_h = F.sigmoid(F.linear(v,self.W,self.h_bias))
        sample_h = self.sample(p_h)
        return p_h,sample_h
    
    def h2v(self,h):
        p_v = F.sigmoid(F.linear(h,self.W.t(),self.v_bias))
        sample_v = self.sample(p_v)
        return p_v,sample_v
    
    def forward(self,v):
        p_h,sample_h = self.v2h(v)

        for _ in range(self.k):
            p_v,sample_v = self.h2v(sample_h)
            p_h,sample_h = self.v2h(sample_v)
        
        return sample_v
    
    def energy(self,v):
        tmp1 = v.mv(self.v_bias)
        tmp2 = torch.sum(F.softplus(F.linear(v,self.W,self.h_bias)),dim=1)
        return torch.mean(-(tmp1+tmp2))

root_dir = "E:\\dl-dataset"
batch_size = 64
epoch_sum = 20

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root=root_dir,train=True,transform=transform,download=False)
test_dataset = datasets.MNIST(root=root_dir,train=False,transform=transform,download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=64,shuffle=True,num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=0)


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = RBM(k=1,device=device)
    net.to(device)
    optimizer = optim.SGD(net.parameters(),lr=0.02,momentum=0.9)
    loss_func = nn.MSELoss()

    for epoch in range(epoch_sum):
        loss_sum = 0
        for i,data in enumerate(train_loader,0):
            optimizer.zero_grad()
            imgs = data[0].to(device)
            imgs = imgs.view(-1,784)
            out = net(imgs)
            energy1 = net.energy(imgs)
            energy2 = net.energy(out)
            loss = energy1-energy2
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            if i%100 ==99:
                print("[%d ,%d]  loss:%.5f"%(epoch,i,loss_sum/100))
                loss_sum = 0
    torch.save(net.state_dict(),"./net.pkl")

train()
net = RBM()
net.load_state_dict(torch.load("./net.pkl"))

def showimg(imgs):
    if torch.is_tensor(imgs):
        imgs = imgs.numpy()
    imgs = np.transpose(imgs,(1,2,0))
    plt.imshow(imgs)
    plt.axis('off')

it = iter(test_loader)
imgs,labels = it.next()
imgs1 = torchvision.utils.make_grid(imgs)
plt.subplot(1,2,1)
showimg(imgs1)



imgs = imgs.view(-1,784)
out = net(imgs)
imgs1 = torchvision.utils.make_grid(out.view(-1,1,28,28).detach())
plt.subplot(1,2,2)
showimg(imgs1)
plt.show()

