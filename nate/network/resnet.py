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

class ResNet50(nn.Module):

    def __init__(self, classCount, isTrained, activation):
        """
        Initialize the resnet50 network
        :param classCount: dimension of the output vector
        :param isTrained: if True then loads pre-trained weights
        :param activation: type of the activation function: ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX
        """

        super(ResNet50, self).__init__()

        self.classcount = classCount

        self.resnet50 = torchvision.models.resnet50(pretrained=isTrained)

        if (activation == ACTIVATION_SIGMOID):
            self.resnet50.fc = nn.Sequential(nn.Linear(512 * 4, classCount), nn.Sigmoid())
        elif (activation == ACTIVATION_SOFTMAX):
            self.resnet50.fc = nn.Sequential(nn.Linear(512 * 4, classCount), nn.Softmax())
        else:
            self.resnet50.fc = nn.Linear(512 * 4, classCount)

    def forward(self, x):
        x = self.resnet50(x)
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

class ResNet101(nn.Module):

    def __init__(self, classCount, isTrained, activation):
        """
        Initialize the resnet101 network
        :param classCount: dimension of the output vector
        :param isTrained: if True then loads pre-trained weights
        :param activation: type of the activation function: ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX
        """

        super(ResNet101, self).__init__()

        self.classcount = classCount

        self.resnet101 = torchvision.models.resnet101(pretrained=isTrained)

        if (activation == ACTIVATION_SIGMOID):
            self.resnet101.fc = nn.Sequential(nn.Linear(512 * 4, classCount), nn.Sigmoid())
        elif (activation == ACTIVATION_SOFTMAX):
            self.resnet101.fc = nn.Sequential(nn.Linear(512 * 4, classCount), nn.Softmax())
        else:
            self.resnet101.fc = nn.Linear(512 * 4, classCount)

    def forward(self, x):
        x = self.resnet101(x)
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

