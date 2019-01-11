import torch.nn as nn

#--------------------------------------------------------------------
#----- This is a shallow network model, that consists of 12 convolutions
#----- with batch normalization, 6 pooling layers.
#----- Author: Manalo M., Andrey G.
#--------------------------------------------------------------------

class Unit(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Basic convolution plus batch normalization unit
        :param in_channels: number of input channels in the unit
        :param out_channels: number of output channels in the unit
        """
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output

#--------------------------------------------------------------------

class ConvNet12(nn.Module):

    def __init__(self, num_classes=1):
        """
        Initialize then network architecture: 12 convolutions, 6 pooling layers
        :param num_classes: number of classes
        """
        super(ConvNet12, self).__init__()

        self.classcount = num_classes

        self.unit1 = Unit(in_channels=3, out_channels=32)
        self.unit2 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit3 = Unit(in_channels=32, out_channels=32)
        self.unit4 = Unit(in_channels=32, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit5 = Unit(in_channels=64, out_channels=64)
        self.unit6 = Unit(in_channels=64, out_channels=64)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit7 = Unit(in_channels=64, out_channels=128)
        self.unit8 = Unit(in_channels=128, out_channels=128)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.unit9 = Unit(in_channels=128, out_channels=256)
        self.unit10 = Unit(in_channels=256, out_channels=256)

        self.pool5 = nn.MaxPool2d(kernel_size=2)


        self.unit11 = Unit(in_channels=256, out_channels=512)
        self.unit12 = Unit(in_channels=512, out_channels=512)

        self.avgpool = nn.AvgPool2d(kernel_size=4)

        self.net = nn.Sequential(self.unit1, self.unit2, self.pool1, self.unit3, self.unit4, self.pool2, self.unit5,
                                 self.unit6, self.pool3, self.unit7, self.unit8, self.pool4, self.unit9, self.unit10,
                                 self.pool5, self.unit11, self.unit12, self.avgpool)

        self.fc = nn.Linear(in_features=512, out_features=num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1,512)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output

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
