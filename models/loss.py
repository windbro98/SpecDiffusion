import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import lpips

def mse_loss(output, target):
    return F.mse_loss(output, target)
    
def l1_loss(output, target):
    criterion = nn.L1Loss(size_average=True, reduce=True)
    return criterion(output, target)

