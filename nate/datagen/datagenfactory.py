from .classification import DatagenClassification
from .segmentation import DatagenAutoencoder
from .segmentation import DatagenSegmentation

# --------------------------------------------------------------------------------
# ---- A factory for creating data generators of different types
# ---- Author: Andrey G.
# --------------------------------------------------------------------------------

class DatagenFactory():

    # ---- Data generator for classification input=image, output=vector
    DATAGEN_CLASSIFICATION = "CLASS"

    # ---- Data generator for image2image training input=image, output=image
    DATAGEN_AUTOENCODER = "IMG2IMG"

    # ---- Data generator for segmentation end transform
    DATAGEN_SEGMENTATION = "SEG"

    def getDatagen(generatorType, setPathImgDir, setPathDataset, setArgs):
        """
        Generates a datat generator of the specified type
        CLASS - datagenerator for classification (setArgs[0]=img transform)
        IMG2IMG - datagenerator for segmentation (setArgs[0]=input img transform, setArgs[1]=output img transform)
        :param setPathImgDir: path to the directory that conatins images for training
        :param setPathDataset: path to the dataset file
        :param setArgs: array of additional arguments
        :return: generator of the specified type or None if can't create
        """
        if not isinstance(setArgs, list): return None

        # -------------------- CLASSIFICATION --------------------
        if generatorType.upper() == DatagenFactory.DATAGEN_CLASSIFICATION:
            if len(setArgs) < 1: return None
            return DatagenClassification(setPathImgDir, setPathDataset, setArgs[0])

        # ---------------------- IMG2IMG ------------------------
        if generatorType.upper() == DatagenFactory.DATAGEN_AUTOENCODER:
            if len(setArgs) < 2: return None
            return DatagenAutoencoder(setPathImgDir, setPathDataset, setArgs[0], setArgs[1])

        # ---------------------- SEGMENTATION --------------------
        if generatorType.upper() == DatagenFactory.DATAGEN_SEGMENTATION:
            if len(setArgs) < 2: return None
            return DatagenSegmentation(setPathImgDir, setPathDataset, setArgs[0], setArgs[1])

        # ---- Datagenerator type is not defined
        return None