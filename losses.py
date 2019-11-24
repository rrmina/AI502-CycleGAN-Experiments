import torch
import torch.nn as nn

def real_loss(x):
    # criterion = nn.MSELoss().to(device)
    # return criterion(x, 1)
    return torch.mean((x-1)**2)

def fake_loss(x):
    # criterion = nn.MSELoss().to(device)
    # return criterion(x, 0)
    return torch.mean(x**2)

def cycle_loss(out, target):
    # criterion = nn.L1Loss().to(device)
    #return criterion(out, target)
    return torch.mean(torch.abs(out-target))