import torch

def dice_coeff(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    https://github.com/pytorch/pytorch/issues/1249
    in V net paper: https://campar.in.tum.de/pub/milletari2016Vnet/milletari2016Vnet.pdf they square the sum in the denominator
    """
    smooth = 1.
    epsilon = 10e-8
    # have to use contiguous since they may from a torch.view op
    iflat = pred.view(-1).contiguous()
    tflat = target.view(-1).contiguous()
    intersection = (iflat * tflat).sum()
    #A_sum = torch.sum(iflat * iflat) #original version from AF
    #B_sum = torch.sum(tflat * tflat) #original version from AF
    A_sum = torch.sum(iflat)
    B_sum = torch.sum(tflat)
    dice = (2. * intersection + smooth) / (A_sum + B_sum + smooth)
    #dice = dice.mean(dim=0)
    #dice = torch.clamp(dice, 0, 1.0)
    return  dice
