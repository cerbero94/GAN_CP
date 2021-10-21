"""
Losses
"""

import torch
import numpy as np
mse = torch.nn.MSELoss(reduction='sum')

# def l2_loss(x,x_hat):
#     """ L2 Loss 

#     Args:
#         x (FloatTensor): Input tensor
#         x_hat (FloatTensor): Output tensor

#     Returns:
#         [FloatTensor]: L2 distance between input and output
#     """
#     return torch.sqrt(torch.sum((x - x_hat)**2))
def l2_loss(x,x_hat):
    return mse(x,x_hat)

def l2_loss_batch(x,xhat):
    losses_array = []
    for i in range(len(x)):
        losses_array.append(np.sqrt(mse(x[i,:],xhat[i,:]).item()))
    return np.array(losses_array)