import torch

from .losszoo import WeightedBinaryCrossEntropy
from .losszoo import WeightedBinaryCrossEntropyMC


class LossFactory():

    LOSS_BCE = "BCE"
    LOSS_WBCE = "WBCE"
    LOSS_WBCEMC = "WBCEMC"
    LOSS_MSE = "MSE"

    def getLossFunction (lossName, weights = None):

        # ---- PyToch MSE loss
        if lossName == LossFactory.LOSS_MSE:
            return torch.nn.MSELoss(size_average=True)

        # ---- Binary cross entropy loss for binary classification
        if lossName == LossFactory.LOSS_BCE:
            return torch.nn.BCELoss(size_average=True)

        # ---- Weighted binary cross entropy for binary classification with unbalanced datasets
        if lossName == LossFactory.LOSS_WBCE:
            if weights != None:
                return WeightedBinaryCrossEntropy(weights[0], 1 - weights[0])
            else:
                return None

        # ---- Weighted binary cross entropy for multi-class classification
        if lossName == LossFactory.LOSS_WBCEMC:
            if weights != None:

                return WeightedBinaryCrossEntropyMC(weights)
            else:
                return None

        return None