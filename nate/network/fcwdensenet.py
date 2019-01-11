import torch
import torch.nn as nn

from .denseteramisu import FCDenseNet, FCDenseNet103
from .denseteramisu.utils import count_parameters, count_conv2d

# --------------------------------------------------------------------------------
# ---- This is a DenseNet Teramisu (FC-DenseNet) wrapper for segmentation
# ---- Author: Andrey G.
# --------------------------------------------------------------------------------

class FCWDenseNet103(nn.Module):

    def __init__(self, classCount):
        """
        This is a FC DenseNet wrapper (DenseNet Teramisu)
        This network accepts tensors of any resolution, but 3 channels (customizable if needed)
        The output is the tensor of the same resolution and classCount channels
        Can be used as an autoencoder if classCount = 3
        :param classCount: number of target segmentation classes
        """

        super(FCWDenseNet103, self).__init__()

        self.classcount = classCount

        self.fcdensenet = FCDenseNet103(in_channels = 3, out_channels=classCount)

    def forward(self, x):
        return self.fcdensenet.forward(x)

    def getsizein(self):
        """
        :return: returns expected dimensions of the input tensor
        """
        return (-1, -1, 3)

    def getsizeout(self):
        """
        :return: returns expected dimensions of the output tensor
        """
        return (-1, -1, self.classcount)

class FCWDenseNet50(nn.Module):

    def __init__(self, classCount):
        """
        This is a FC DenseNet wrapper (DenseNet Teramisu)
        This network accepts tensors of any resolution, but 3 channels (customizable if needed)
        The output is the tensor of the same resolution and classCount channels
        Can be used as an autoencoder if classCount = 3
        :param classCount: number of target segmentation classes
        """
        super(FCWDenseNet50, self).__init__()
        self.classcount = classCount

        self.fcdensenet = FCDenseNet(
            in_channels=3,
            out_channels=self.classcount,
            initial_num_features=24,
            dropout=0.2,

            down_dense_growth_rates=8,
            down_dense_bottleneck_ratios=None,
            down_dense_num_layers=(4, 5, 7),
            down_transition_compression_factors=1.0,

            middle_dense_growth_rate=8,
            middle_dense_bottleneck=None,
            middle_dense_num_layers=10,

            up_dense_growth_rates=8,
            up_dense_bottleneck_ratios=None,
            up_dense_num_layers=(7, 5, 4)
        )

    def forward(self, x):
        return self.fcdensenet.forward(x)

    def getsizein(self):
        """
        :return: returns expected dimensions of the input tensor
        """
        return (-1, -1, 3)

    def getsizeout(self):
        """
        :return: returns expected dimensions of the output tensor
        """
        return (-1, -1, self.classcount)
