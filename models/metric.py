import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchmetrics.functional.regression import pearson_corrcoef, mean_squared_error
from torchmetrics.functional.image import structural_similarity_index_measure, peak_signal_noise_ratio
from torch.nn.functional import l1_loss

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

def mae(target, input):
    with torch.no_grad():
        output = l1_loss(input, target)
    return output

def mse(target, input):
    with torch.no_grad():
        output = mean_squared_error(input, target)
    return output

def psnr(target, input):
    with torch.no_grad():
        output = peak_signal_noise_ratio(input, target)
    return output

def ssim(target, input):
    with torch.no_grad():
        output = structural_similarity_index_measure(input, target)
    return output

def pearson(target, input):
    n = target.shape[0]
    target = target.reshape(-1, n)
    input = input.reshape(-1, n)
    # num_outputs = target.shape[1]
    with torch.no_grad():
        output = torch.mean(pearson_corrcoef(target, input))
    return output