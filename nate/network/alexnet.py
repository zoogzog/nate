import torch
import torch.nn as nn
import torchvision

# --------------------------------------------------------------------------------
# ---- Collection of neural network architectures with AlexNet base
# ---- Author: Andrey G.
# --------------------------------------------------------------------------------

ACTIVATION_SIGMOID = 0
ACTIVATION_SOFTMAX = 1
ACTIVATION_NONE = 2

# --------------------------------------------------------------------------------

class AlexNet(nn.Module):

    def __init__(self, classCount, isTrained, activation):
        """
        Initialize the alexnet network
        :param classCount: dimension of the output vector
        :param isTrained: if True then loads pre-trained weights
        :param activation: type of the activation function: ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX
        """

        self.classcount = classCount

        super(AlexNet, self).__init__()

        self.alexnet = torchvision.models.alexnet(pretrained=isTrained)

        if (activation == ACTIVATION_SIGMOID):
            self.alexnet.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, classCount),
                nn.Sigmoid()
            )
        elif (activation == ACTIVATION_SOFTMAX):
            self.alexnet.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, classCount),
                nn.Softmax
            )
        else:
            self.alexnet.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, classCount),
            )

    def forward(self, x):
        x = self.alexnet(x)
        return x

    def getsizein(self):
        return (224, 224, 3)

    def getsizeout(self):
        return self.classcount

# --------------------------------------------------------------------------------