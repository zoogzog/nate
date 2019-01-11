import os
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# --------------------------------------------------------------------------------
# ---- This is a dataset generator class that extends the basic pytorch dataset generator
# ---- so that instead of providing a directory with images we can provide a text file with
# ---- a specific structure of the dataset. Each row in the file contains path to the input
# ---- file and path to the output image file.
# ---- <row> = <input image path> <output image path>
# ---- Author: Andrey G.
# --------------------------------------------------------------------------------

class DatagenAutoencoder(Dataset):

    # ------------------------------------ PRIVATE -----------------------------------
    def __init__(self, pathimgdir, pathdataset, transformin, transformout):
        """
        :param pathimgdir: path to the directory that contains images
        :param pathdataset: path to the file of the dataset
        :param transformin: transform function for the input images
        :param transformout: transform functio for the output image
        """

        self.listimginput = []
        self.listimgoutput = []
        self.transformin = transformin
        self.transformout = transformout

        # ---- Open the dataset file
        filedescriptor = open(pathdataset, "r")

        # ---- Scan the file and save into the internal class storage
        line = True

        while line:

            line = filedescriptor.readline()

            # --- if the line is not empty - then process it
            if line:
                lineitems = line.split()

                imagepathin = os.path.join(pathimgdir, lineitems[0])
                imagepathout = os.path.join(pathimgdir, lineitems[1])

                self.listimginput.append(imagepathin)
                self.listimgoutput.append(imagepathout)

        filedescriptor.close()

    # --------------------------------------------------------------------------------

    def __getitem__(self, index):

        imgpathin = self.listimginput[index]
        imgpathout = self.listimgoutput[index]

        datain = Image.open(imgpathin).convert('RGB')
        dataout = Image.open(imgpathout).convert('RGB')

        datain = self.transformin(datain)
        dataout = self.transformout(dataout)

        return datain, dataout

    # --------------------------------------------------------------------------------

    def __len__(self):
        return len(self.listimginput)

    # --------------------------------------------------------------------------------

    def getsize(self):
        """
        Get the number of samples in the dataset
        :return: (int) - size of the dataset
        """
        return len(self.listimginput)

    def getweights(self):
        return None

    def getfrequency(self):
        return (1,1)

# --------------------------------------------------------------------------------
# ---- This is a dataset generator class that extends the basic pytorch dataset generator
# ---- so that instead of providing a directory with images we can provide a text file with
# ---- a specific structure of the dataset. This generator works for segmentation task
# ---- datasets. Since the segmentation data is provided in an image format a label map
# ---- should be stored in the dataset file.
# ------------------------------------
# ---- dict: <r> <g> <b> <segment_ID_1>
# ---- dict: <r> <g> <b> <segment_ID_2>
# ---- .....
# ---- <input image path> <output image path>
# ------------------------------------
# --------------------------------------------------------------------------------


class DatagenSegmentation(Dataset):
    def __init__(self, pathimgdir, pathdataset, transformin, transformout):
        """
        :param pathimgdir: path to the directory that contains images
        :param pathdataset: path to the file of the dataset
        :param transformin: transform function for the input images
        :param transformout: transform functio for the output image
        """
        # ---- This is a list of input file paths
        self.listimginput = []
        # ---- This is a list of output target maps
        self.listimgoutput = []
        # ---- This is a list of conversions
        self.lutForward = {}
        self.lutBackward = {}

        self.transformin = transformin
        self.transformout = transformout

        # ---- Open the dataset file
        filedescriptor = open(pathdataset, "r")

        # ---- Scan the file and save into the internal class storage
        line = True

        while line:

            line = filedescriptor.readline()

            # ---- if the line is not empty - then process it
            if line:
                lineitems = line.split()

                # ---- Wrong format
                if len(lineitems) < 2: return

                # ---- If it is a dictionary item add to the dictionary
                if lineitems[0] == "dict:":
                    if len(lineitems) == 5:
                        r = lineitems[1]
                        g = lineitems[2]
                        b = lineitems[3]
                        labelID = lineitems[4]

                        key = r + "-" + g + "-" + b

                        self.lutForward[key] = int(labelID)
                        self.lutBackward[int(labelID)] = (r, g, b)

                else:
                    imagepathin = os.path.join(pathimgdir, lineitems[0])
                    imagepathout = os.path.join(pathimgdir, lineitems[1])

                    self.listimginput.append(imagepathin)
                    self.listimgoutput.append(imagepathout)

        print(self.lutBackward)

        filedescriptor.close()

    def __len__(self):
        return len(self.listimginput)

    def __getitem__(self, index):

        imgpathin = self.listimginput[index]
        imgpathout = self.listimgoutput[index]

        datain = Image.open(imgpathin).convert('RGB')
        dataout_raw = Image.open(imgpathout).convert('RGB')

        # ---- For the input image the transformation is done directly into tensor
        datain = self.transformin(datain)

        # ---- For the output image the transformation is only crops, need to apply map, normalize, toTensor
        dataout = self.transformout(dataout_raw)

        # ---- We generated the proper map, convert it to tensor
        dataout_map = self.img2map(dataout)

        dataout_map = np.moveaxis(dataout_map , -1, 0)
        dataout_map = torch.from_numpy(dataout_map)
        dataout_map = dataout_map.float()

        return datain, dataout_map

    # ----- Convert image (PIL) into map representation
    def img2map (self, image):
        """
        :param image: PIL image of size WxHx3
        :return: map of size WxHxN
        """
        mapData = np.array(image)

        imgHeight = mapData.shape[0]
        imgWidth = mapData.shape[1]
        imgChannel = mapData.shape[2]

        size = (imgHeight, imgWidth, len(self.lutForward))
        mapx = np.zeros(size)

        for i in range (0, imgHeight):
            for j in range (0, imgWidth):
                color = mapData[i][j]
                colorName = str(color[0]) + "-" + str(color[1]) + "-" + str(color[2])

                if colorName in self.lutForward:
                    label = self.lutForward[colorName]

                    #---- we assume that labe
                    # l mapping starts from 1
                    mapx[i][j][label - 1] = 1

        return mapx

    # ----- Convert a map into an image (PIL)
    def map2img(self, map):
        """
        :param map: MAP - WxHxN map converted from segmentation
        :return:  a PIL image
        """

        mapData = map

        imgHeight = mapData.shape[0]
        imgWidth = mapData.shape[1]
        imgChannel = mapData.shape[2]

        size = (imgHeight, imgWidth, 3)

        imgout = np.zeros(size)

        for i in range (0, imgHeight):
            for j in range (0, imgWidth):
                arr = np.nonzero(mapData[i][j] == 1)

                if len(arr[0]) != 0:
                    index = arr[0][0]
                    #print(index)
                    imgout[i][j] = self.lutBackward[index + 1]

        imgout = np.uint8(imgout)
        imgout = Image.fromarray(imgout)

        return imgout

    def getsize(self):
        """
        Get the number of samples in the dataset
        :return: (int) - size of the dataset
        """
        return len(self.listimginput)

    def getfrequency(self):
        return (1,1)