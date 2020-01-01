import os
import torch
import time
import numpy as np
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torchvision.utils import save_image
#import visdom
import matplotlib.pyplot as plt

from model import VAE, loss_function, weigth_init
from util import showimg

epoch_sum = 10
lr = 1e-6

root_dir = 'E:\\dl-dataset\\cifar-10-python'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img.float()),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])



train_dataset = datasets.CIFAR10(root=root_dir, train=True, transform=transform, download=False)
test_dataset = datasets.CIFAR10(root=root_dir,train=False,transform=transform,download=False)

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=0)

test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=64,
                                              shuffle=False,
                                              num_workers=0)
                    
def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vae = VAE(device)
    vae.to(device)
    optimizer = optim.SGD(vae.parameters(), lr=lr)
    #viz = visdom.Visdom()
    print(time.asctime(time.localtime(time.time())))
    n = 0
    for epoch in range(epoch_sum):
        loss_sum = 0
        for idx, image in enumerate(train_dataloader,0):
            image = image[0].to(device)
            optimizer.zero_grad()
            out, mu, logvar = vae(image)
            loss = loss_function(out, image, mu, logvar)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
            #viz.line([loss.item()], [n], update="append", win='loss_win')
            n += 1
            if idx % 100 == 99:
                print("epoch:%d, idx:%d, loss:%.6f" % (epoch + 1, idx, loss_sum / 100))
                loss_sum = 0
        torch.save(vae.state_dict(), './vae.pth')
    print(time.asctime(time.localtime(time.time())))

def show_reconstruction():
    vae = VAE()
    vae.load_state_dict(torch.load("./vae.pth"))
    it = iter(test_dataloader)
    imgs,_ = it.next()
    imgs1 = torchvision.utils.make_grid(imgs)
    plt.subplot(1,2,1)
    showimg(imgs1)
    
    out,_,_ = vae(imgs)
    out = torchvision.utils.make_grid(out)
    plt.subplot(1,2,2)
    showimg(out.detach())
    plt.show()

#train()
show_reconstruction()