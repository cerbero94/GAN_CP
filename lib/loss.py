"""
Losses
"""

import torch
import numpy as np

mse = torch.nn.MSELoss(reduction='none')

def l2_loss(x,x_hat):
    loss = mse(x,x_hat)
    loss = torch.sum(loss, dim=1) 
    loss = torch.mean(torch.sqrt(loss))
    return loss

def l2_loss_batch(x,x_hat):
    loss = mse(x,x_hat)
    loss = torch.sum(loss, dim=1) 
    loss = torch.sqrt(loss)
    return loss.numpy()
