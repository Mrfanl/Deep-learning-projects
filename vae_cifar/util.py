import numpy as np
import torch
import matplotlib.pyplot as plt

def showimg(imgs):
    if torch.is_tensor(imgs):
        imgs = imgs.numpy()
    imgs = np.transpose(imgs,(1,2,0))
    plt.imshow(imgs)
    plt.axis("off")


    

