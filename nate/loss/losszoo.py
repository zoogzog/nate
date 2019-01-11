import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------------------------------------------------------------------------------
# ---- USED IN: Binary classification tasks with unbalanced datasets
class WeightedBinaryCrossEntropy(torch.nn.Module):

    def __init__(self, weightPOS, weightNEG):
        super(WeightedBinaryCrossEntropy, self).__init__()

        self.wp = weightPOS
        self.wn = weightNEG

    def forward(self, output, target):
        wp = self.wp
        wn = self.wn

        loss = -(
            (wn * (target * torch.log(output + 0.00001)) + wp * ((1 - target) * torch.log(1 - output + 0.00001))).sum(
                1).mean())

        return loss

# --------------------------------------------------------------------------------
# ----- Loss: weighted binary cross entropy (multiclass version)

def multiclass_binary_cross_entropy(prediction, target, weight=None):
    # Bunch of assertions to make sure what we got is good, yay Pythonnnnn typeless poooooop
    assert torch.is_tensor(prediction)
    assert torch.is_tensor(target)
    assert torch.is_tensor(weight)
    # Checking sizes
    assert prediction.size() == target.size()
    size = target.size()
    L  = F.binary_cross_entropy(prediction, target, weight=weight, size_average=False)

    return torch.mean(L)


class WeightedBinaryCrossEntropyMC(torch.nn.Module):

    def __init__(self, weights):

        super(WeightedBinaryCrossEntropyMC, self).__init__()

        self.weights_zero = np.array(weights)
        self.weights_one = 1 - np.array(weights)

    def forward (self, output, target):
        gt = target.cpu().numpy()

        wt = self.weights_one * gt + (1 - gt) * self.weights_zero

        wt = torch.from_numpy(wt).type(torch.FloatTensor).cuda()

        L = multiclass_binary_cross_entropy(output, target, wt)

        return L