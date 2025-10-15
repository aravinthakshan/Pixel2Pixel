import torch.nn as nn
import torch

def mse_loss(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return nn.MSELoss()(gt, pred)

def loss_func(model, img1, img2, loss_f=nn.MSELoss()):
    pred1 = model(img1)
    loss = loss_f(img2, pred1)
    return loss
