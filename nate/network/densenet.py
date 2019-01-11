import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

# --------------------------------------------------------------------------------
# ---- Collection of neural network architectures with DenseNetBase
# ---- Author: Andrey G.
# --------------------------------------------------------------------------------

ACTIVATION_SIGMOID = 0
ACTIVATION_SOFTMAX = 1
ACTIVATION_NONE = 2

# --------------------------------------------------------------------------------

class DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained, activation):
        """
        Initialize the densenet121 network
        :param classCount: dimension of the output vector
        :param isTrained: if True then loads pre-trained weights
        :param activation: type of the activation function: ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX
        """

        super(DenseNet121, self).__init__()

        self.classcount = classCount

        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)

        kernelCount = self.densenet121.classifier.in_features

        if (activation == ACTIVATION_SIGMOID):
            self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        elif (activation == ACTIVATION_SOFTMAX):
            self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Softmax())
        else:
            self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount))

    def forward(self, x):
        x = self.densenet121(x)
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

class DenseNet169(nn.Module):

    def __init__(self, classCount, isTrained, activation):
        """
        Initialize the densenet169 network
        :param classCount: dimension of the output vector
        :param isTrained: if True then loads pre-trained weights
        :param activation: type of the activation function: ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX
        """
        super(DenseNet169, self).__init__()

        self.classcount = classCount

        self.densenet169 = torchvision.models.densenet169(pretrained=isTrained)

        kernelCount = self.densenet169.classifier.in_features

        if (activation == ACTIVATION_SIGMOID):
            self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        elif (activation == ACTIVATION_SOFTMAX):
            self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Softmax())
        else:
            self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, classCount))

    def forward(self, x):
        x = self.densenet169(x)
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

class DenseNet201(nn.Module):

    def __init__(self, classCount, isTrained, activation):
        """
        Initialize the densenet201 network
        :param classCount: dimension of the output vector
        :param isTrained: if True then loads pre-trained weights
        :param activation: type of the activation function: ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX
        """

        super(DenseNet201, self).__init__()

        self.classcount = classCount

        self.densenet201 = torchvision.models.densenet201(pretrained=isTrained)

        kernelCount = self.densenet201.classifier.in_features

        if (activation == ACTIVATION_SIGMOID):
            self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        elif (activation == ACTIVATION_SOFTMAX):
            self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Softmax())
        else:
            self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount, classCount))

    def forward(self, x):
        x = self.densenet201(x)
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

class HRDenseNet121(nn.Module):

    def __init__(self, classCount, isTrained, activation):
        """
        This network supports images with arbitrary resolution (should be a square), the backbone is densenet121
        :param classCount: dimension of the output vector / number of classes
        :param isTrained: if True then loads pre-trained weights
        :param activation: type of the activation function: ACTIVATION_SIGMOID or ACTIVATION_SOFTMAX
        """
        super(HRDenseNet121, self).__init__()

        self.classcount = classCount

        self.densenet = torchvision.models.densenet121(pretrained=isTrained)

        self.features = self.densenet.features
        self.classifier = self.densenet.classifier

        kernelCount = self.classifier.in_features

        if (activation == ACTIVATION_SIGMOID):
            self.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        elif (activation == ACTIVATION_SOFTMAX):
            self.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Softmax())
        else:
            self.classifier = nn.Sequential(nn.Linear(kernelCount, classCount))

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=features.size(-1), stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out

    def getsizein(self):
        """
        :return: returns expected dimensions of the input tensor
        """
        return (-1, -1, 3)

    def getsizeout(self):
        """
        :return: returns expected dimensions of the output tensor
        """
        return self.classcount

# --------------------------------------------------------------------------------

class HRDenseNet169(nn.Module):

    def __init__(self, classCount, isTrained, activation):
        """
        This network supports images with arbitrary resolution (should be a square), the backbone is densenet169
        :param classCount: dimension of the output vector / number of classes
        :param isTrained: if True then loads pre-trained weights
        :param activation: type of the activation function: ACTIVATION_SIGMOID or ACTIVATION_SOFTMAX
        """
        super(HRDenseNet169, self).__init__()

        self.classcount = classCount

        self.densenet = torchvision.models.densenet169(pretrained=isTrained)

        self.features = self.densenet.features
        self.classifier = self.densenet.classifier

        kernelCount = self.classifier.in_features

        if (activation == ACTIVATION_SIGMOID):
            self.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        elif (activation == ACTIVATION_SOFTMAX):
            self.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Softmax())
        else:
            self.classifier = nn.Sequential(nn.Linear(kernelCount, classCount))

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=features.size(-1), stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out

    def getsizein(self):
        """
        :return: returns expected dimensions of the input tensor
        """
        return (-1, -1, 3)

    def getsizeout(self):
        """
        :return: returns expected dimensions of the output tensor
        """
        return self.classcount
