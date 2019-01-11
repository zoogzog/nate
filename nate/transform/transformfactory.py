import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

class TransformFactory():

    TRANSFORM_RNDCROP = "RNDCROP"
    TRANSFORM_RESIZE = "RESIZE"
    TRANSFORM_CCROP = "CCROP"
    TRANSFORM_10CROP = "10CROP"

    TRANSFORM_TABLE =\
        {
            TRANSFORM_RNDCROP : 0,
            TRANSFORM_RESIZE : 1,
            TRANSFORM_CCROP : 2,
            TRANSFORM_10CROP : 3
        }

    def getTransform(transformName, transformArg):
        if transformName.upper() == TransformFactory.TRANSFORM_RNDCROP:
            return transforms.RandomResizedCrop(transformArg)
        if transformName.upper() == TransformFactory.TRANSFORM_RESIZE:
            return transforms.Resize((transformArg, transformArg))
        if transformName.upper() == TransformFactory.TRANSFORM_CCROP:
            return transforms.CenterCrop(transformArg)
        if transformName.upper() == TransformFactory.TRANSFORM_10CROP:
            return transforms.TenCrop(transformArg)

        return None

    def getTransformMap(transformName, transformArg):
        if transformName.upper() == TransformFactory.TRANSFORM_RNDCROP:
            return transforms.RandomResizedCrop(transformArg, interpolation=Image.NEAREST)
        if transformName.upper() == TransformFactory.TRANSFORM_RESIZE:
            return transforms.Resize((transformArg, transformArg), interpolation=Image.NEAREST)
        if transformName.upper() == TransformFactory.TRANSFORM_CCROP:
            return transforms.CenterCrop(transformArg)

        return None

    def getTransformSequence(trSequence, trSequenceArgs, isNormalize = True):
        """
         Get the transformation sequence for the train/val/test
         :return: (transformation sequence) structure to be used by the loader
         """
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []

        istencrop = False

        for i in range(0, len(trSequence)):
            transformName = trSequence[i]
            transformArg = trSequenceArgs[i]

            transformG = TransformFactory.getTransform(transformName, transformArg)
            #----- EXCEPTION - CANT GENERATE TRANSFORMATION
            if transformG == None: return None

            transformList.append(transformG)

            if transformName.upper() == TransformFactory.TRANSFORM_10CROP: istencrop = True

        #----- Generate 10 crop
        if istencrop:
            transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
            if isNormalize: transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        else:
            transformList.append(transforms.ToTensor())
            if isNormalize:
                transformList.append(normalize)

        transformSequence = transforms.Compose(transformList)
        return transformSequence

    def getTransformSequenceMap(trSequence, trSequenceArgs):
        """
         Get the transformation sequence for the train/val/test
         :return: (transformation sequence) structure to be used by the loader
         """
        transformList = []

        # >>>>>>>>>>>>>>>>>
        for i in range(0, len(trSequence)):
            transformName = trSequence[i]
            transformArg = trSequenceArgs[i]

            transformG = TransformFactory.getTransformMap(transformName, transformArg)
            #----- EXCEPTION - CANT GENERATE TRANSFORMATION
            if transformG == None: return None

            transformList.append(transformG)
        # >>>>>>>>>>>>>>>>>

        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def tensor2image(tensor, isReverseNormalize = False):
        """
        Convert a tensor back to an image. If necessary reverse normalization procedure.
        :param tensor: a tensor
        :param isReverseNormalize: set to true to reverse normalization
        :return: image
        """

        if isReverseNormalize:
            inv_normalize = transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
            )

            tensor = inv_normalize(tensor)

        data = tensor.numpy()
        data = np.moveaxis(data, -1, 0)
        data = np.moveaxis(data, -1, 0)
        data = data * 255
        data = 255 - np.uint8(data)

        im = Image.fromarray(data * 255)


        return im



