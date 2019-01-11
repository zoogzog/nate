import torch
import torch.nn as nn
import torchvision

# --------------------------------------------------------------------------------
# ---- Collection of neural network architectures with ResNet base
# ---- Author: Andrey G. + torchvision
# --------------------------------------------------------------------------------

ACTIVATION_SIGMOID = 0
ACTIVATION_SOFTMAX = 1
ACTIVATION_NONE = 2

# --------------------------------------------------------------------------------

class Inception(nn.Module):

    def __init__(self, classCount, isTrained, activation):
        """
        Initialize the Inception network
        :param classCount: dimension of the output vector
        :param isTrained: if True then loads pre-trained weights
        :param activation: type of the activation function: ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX
        """
        super(Inception, self).__init__()

        self.classcount = classCount

        self.inception = torchvision.models.inception_v3(pretrained=isTrained)

        if (activation == ACTIVATION_SIGMOID):
            self.inception.fc = nn.Sequential(nn.Linear(2048, classCount), nn.Sigmoid())
        elif (activation == ACTIVATION_SOFTMAX):
            self.inception.fc = nn.Sequential(nn.Linear(2048, classCount), nn.Softmax())
        else:
            self.inception.fc = nn.Linear(2048, classCount)

    def forward(self, x):
        x = self.inception(x)
        return x


    def getsizein(self):
        """
        :return: returns expected dimensions of the input tensor
        """
        return (224, 224, 3)

    def getsizeout(self):
        """
        :return: returns expected dimensions of the output tensor
        """
        return self.classcount

# --------------------------------------------------------------------------------