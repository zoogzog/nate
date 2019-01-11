from .alexnet import AlexNet
from .convnet import ConvNet12
from .densenet import DenseNet121
from .densenet import DenseNet169
from .densenet import DenseNet201
from .densenet import HRDenseNet121
from .densenet import HRDenseNet169
from .inceptionnet import Inception
from .resnet import ResNet50
from .resnet import ResNet101
from .vggnet import VGGN16
from .scalenet import ScaleNet3
from .fcwdensenet import FCWDenseNet103
from .fcwdensenet import FCWDenseNet50

# --------------------------------------------------------------------------------
# ---- A factory for creating different network architectures
# ---- Author: Andrey G.
# --------------------------------------------------------------------------------

class NetworkFactory():
    """
    A factory class that generates a network model.
    """

    NET_ALEXNET = "ALEXNET"
    NET_CONVNET12 = "CONVNET12"
    NET_DENSENET121 = "DENSENET121"
    NET_DENSENET169 = "DENSENET169"
    NET_DENSENET201 = "DENSENET201"
    NET_INCEPTION = "INCEPTION"
    NET_RESNET50 = "RESNET50"
    NET_RESNET101 = "RESNET101"
    NET_VGGN16 = "VGGN16"
    NET_HRDENSENET121 = "HRDENSENET121"
    NET_HRDENSENET169 = "HRDENSENET169"
    NET_SCALENET3 = "SCALENET3"
    NET_FCDENSENET103 = "FCDENSENET103"
    NET_FCDENSENET50 = "FCDENSENET50"


    NETWORK_TABLE = \
        {
            NET_ALEXNET: 1,
            NET_CONVNET12: 2,
            NET_DENSENET121: 3,
            NET_DENSENET169: 4,
            NET_DENSENET201: 5,
            NET_INCEPTION: 6,
            NET_RESNET50: 7,
            NET_RESNET101: 8,
            NET_VGGN16: 9,
            NET_HRDENSENET121: 10,
            NET_HRDENSENET169: 11,
            NET_SCALENET3: 12,
            NET_FCDENSENET103: 13,
            NET_FCDENSENET50: 14
        }

    def getNetwork (netName, netClassCount = -1, netActivation = 0, netTrained = True):
        """
        Generate a network with a desired architecture
        :param netName: name of the architecture
        ALEXNET, CONVNET12, DENSENET121, DENSENET169, DENSENET201, INCEPTION, RESNET50, RESNET101, VGGN16
        :param netClassCount: number of output classes, or -1 in case of img2img architectures
        :param netActivation: activation function (for densenet, resnet, inception and vggn16)
        :param netTrained: if true then use pre-trained (uses torchvision pre-trained networks)
        :return: network model or None if can't create the network
        """

        if netName in NetworkFactory.NETWORK_TABLE:
            # ----------------------- ALEXNET -----------------------
            if netName == NetworkFactory.NET_ALEXNET:
                if netClassCount == -1: return None
                else: return AlexNet(netClassCount, netTrained, netActivation)
            # ----------------------- CONVNET12 ---------------------
            if netName == NetworkFactory.NET_CONVNET12:
                if netClassCount == -1: return None
                else: return ConvNet12(netClassCount)
            # ---------------------- DENSENET121 --------------------
            if netName == NetworkFactory.NET_DENSENET121:
                if netClassCount == -1: return None
                else: return DenseNet121(netClassCount, netTrained, netActivation)
            # ---------------------- DENSENET169 --------------------
            if netName == NetworkFactory.NET_DENSENET169:
                if netClassCount == -1: return None
                else: return DenseNet169(netClassCount, netTrained, netActivation)
            # ---------------------- DENSENET201 --------------------
            if netName == NetworkFactory.NET_DENSENET201:
                if netClassCount == -1: return None
                else: return DenseNet201(netClassCount, netTrained, netActivation)
            # ---------------------- INCEPTION ----------------------
            if netName == NetworkFactory.NET_INCEPTION:
                if netClassCount == -1: return None
                else: return Inception(netClassCount, netTrained, netActivation)
            # ---------------------- RESNET50 -----------------------
            if netName == NetworkFactory.NET_RESNET50:
                if netClassCount == -1: return None
                else: return ResNet50(netClassCount, netTrained, netActivation)
            # ---------------------- RESNET101 ----------------------
            if netName == NetworkFactory.NET_RESNET101:
                if netClassCount == -1: return None
                else: return ResNet101(netClassCount, netTrained, netActivation)
            # ------------------------ VGGN16 ------------------------
            if netName == NetworkFactory.NET_VGGN16:
                if netClassCount == -1: return None
                else: return VGGN16(netClassCount, netTrained, netActivation)
            # ---------------------- DENSENET121 --------------------
            if netName == NetworkFactory.NET_HRDENSENET121:
                if netClassCount == -1: return None
                else: return HRDenseNet121(netClassCount, netTrained, netActivation)
            # ---------------------- DENSENET169 --------------------
            if netName == NetworkFactory.NET_HRDENSENET169:
                if netClassCount == -1: return None
                else: return HRDenseNet169(netClassCount, netTrained, netActivation)
            # ---------------------- SCALENET3 ----------------------
            if netName == NetworkFactory.NET_SCALENET3:
                if netClassCount == -1: return None
                else: return ScaleNet3(netClassCount)
            # --------------------- FCDENSENET103 -------------------
            if netName == NetworkFactory.NET_FCDENSENET103:
                if netClassCount == -1:
                    return None
                else:
                    return FCWDenseNet103(netClassCount)
            # --------------------- FCDENSENET50 ---------------------
            if netName == NetworkFactory.NET_FCDENSENET50:
                if netClassCount == -1:
                    return None
                else:
                    return FCWDenseNet50(netClassCount)


        #---------- UNKNOWN NETWORK NAME
        else: return None
