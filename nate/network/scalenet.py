import torch
import torch.nn as nn
from collections import OrderedDict

# --------------------------------------------------------------------------------
# ---- Scale net carries out convolutions on different scales and concatenates
# ---- them together to carry out classficiation
# ---- Author: Andrey G.
# ---- * There was some paper related to the coded architecture, but can't find :(
# ---- * I think it was called fusion network
# ---- * I renamed it into ScaleNet3 cause we have 3 instances of scaled maps here
# ---- * Pay attention! Required input is 227x227x3 image
# --------------------------------------------------------------------------------

class ScaleNet3(nn.Module):

    def __init__ (self, classCount):

        super(ScaleNet3,self).__init__()

        self.classcount = classCount

        self.step1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1, bias=False)),
            ('relux', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size = 3, stride = 2)),
            ('relux', nn.ReLU(inplace=True))
        ]))

        self.poolA = nn.Sequential(OrderedDict([
            ('pool11', nn.AvgPool2d(kernel_size = 3, stride = 4, padding=0)),
            ('relux', nn.ReLU(inplace=True))
        ]))

        self.step2 = nn.Sequential(OrderedDict([
            ('norm1', nn.BatchNorm2d(96)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=1)),
            ('relux', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=3, stride=2, padding = 1)),
            ('relux', nn.ReLU(inplace=True))
        ]))

        self.poolB = nn.Sequential(OrderedDict([
            ('pool11', nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 1)),
            ('relux', nn.ReLU(inplace=True))
        ]))

        self.step3 = nn.Sequential(OrderedDict([
            ('norm2', nn.BatchNorm2d(256)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)),
            ('relux', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)),
            ('relux', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=2)),
            ('relux', nn.ReLU(inplace=True)),
            ('pool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('relu5', nn.ReLU(inplace=True))
        ]))

        self.classifier  = nn.Sequential(OrderedDict([
            ('fc6', nn.Linear(608 * 7 * 7 , 4096)),
            ('relux', nn.ReLU(inplace=True)),
            ('fc7', nn.Linear(4096, classCount)),
            ('sigm', nn.Sigmoid())
        ]))

        #---------- INITIALIZE ----------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward (self, x):
        x = self.step1(x)
        poolA = self.poolA(x)
        x = self.step2(x)
        poolB = self.poolB(x)
        x = self.step3(x)

        resW = x.size()[2]
        resH = x.size()[3]

        x = torch.cat([x, poolA, poolB], 1)

        x = x.view(-1, 608 * resH * resW)
        x = self.classifier(x)

        return x

    def getsizein(self):
        """
        :return: returns expected dimensions of the input tensor
        """
        return (227, 227, 3)

    def getsizeout(self):
        """
        :return: returns expected dimensions of the output tensor
        """
        return self.classcount