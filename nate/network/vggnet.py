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

class VGGN16(nn.Module):

    def __init__(self, classCount, isTrained, activation):
        """
        Initialize the vggn16 network
        :param classCount: dimension of the output vector
        :param isTrained: if True then loads pre-trained weights
        :param activation: type of the activation function: ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX
        """
        super(VGGN16, self).__init__()

        self.classcount = classCount

        self.vgg = torchvision.models.vgg16_bn(pretrained=isTrained)

        if (activation == ACTIVATION_SIGMOID):
            self.vgg.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, classCount),
                nn.Sigmoid()
            )
        elif (activation == ACTIVATION_SOFTMAX):
            self.vgg.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, classCount),
                nn.Softmax()
            )
        else:
            self.vgg.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, classCount),
            )

    def forward(self, x):
        x = self.vgg(x)
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