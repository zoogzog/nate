class TaskSettings():

    # ------------------------------
    # -- TASK_TYPE:
    # ------ CLASS - classification
    # ------ IMG2IMG - autoencoder type, img 2 img transfer
    # ------ SEG - classical segmentation img to data
    # -- TASK_STAGE:
    # ------ TRAIN - perform only training
    # ------ TEST - perform only testing (have to specify checkpoint)
    # ------ RESUME - continue training from the checkpoint
    # ------ S2E - end to end from training to testing
    # -- TASK_CHECKPOINT: any string or empty, if empty then generated automatically
    # -- NETWORK_TYPE: - name of a network
    # ------ SUPPORTED: ALEXNET, CONVNET12, DENSENET121, DENSENET169, DENSENET201, INCEPTION, RESNET50, RESNET101,
    # ------ VGGN16, HRDENSENET121, HRDENSENET169, SCALENET3, FCDENSENET103, FCDENSENET50
    # -- NETWORK_ISTRAINED: If True then attempt to use pre-trained network (ImageNet)
    # -- NETWORK_CLASSCOUNT: For classification tasks - dimension of the output vector
    # -- ACTIVATION_TYPE: For some networks - possible to specify activation type
    # ------- 0 - Sigmoid, 1 - Softmax, 2 - None
    # -- LEARNING_RATE: learning rate for the network
    # -- TRANSFORM_SEQUENCE_***, TRANSFORM_SEQUENCE_***_PARAMETERS - transformation types and its arguments for ***
    # ------- SUPPORTED: RNDCROP, RESIZE, CCROP, 10CROP
    # -- LOSS: BCE, WBCE, WBCEMC
    # -- EPOCH: number of epochs to carry out training
    # -- BATCH: size of the batch to use
    # -- PATH_IN_ROOT - path to the database with images
    # -- PATH_IN_TRAIN, PATH_IN_VALIDATE, PATH_IN_TEST - path to the training, validation and testing dataset files
    # -------- Can be omitted for specific tasks
    # -- PATH_OUT_LOG - directory where the execution log will be saved
    # -- PATH_OUT_MODEL - directory where the trained model will be saved
    # -- PATH_OUT_ACCURACY - directory where the file with accuracy scored will be saved
    # ------------------------------

    TASK_TYPE_CLASS = "CLASS"
    TASK_TYPE_IMG2IMG = "IMG2IMG"
    TASK_TYPE_SEG = "SEG"

    TASK_STAGE_TRAIN = "TRAIN"
    TASK_STAGE_TEST = "TEST"
    TASK_STAGE_RESUME = "RESUME"
    TASK_STAGE_S2E = "S2E"

    TASK_TYPE_LIST = {TASK_TYPE_CLASS, TASK_TYPE_IMG2IMG, TASK_TYPE_SEG}
    TASK_STAGE_LIST = {TASK_STAGE_TRAIN, TASK_STAGE_TEST, TASK_STAGE_RESUME, TASK_STAGE_S2E}

    # --------------------------------------------------------------------------------
    def __init__(self):
        self.TASK_TYPE = ""
        self.TASK_STAGE = ""
        self.TASK_CHECKPOINT = ""

        self.PATH_IN_ROOT = ""
        self.PATH_IN_TRAIN = ""
        self.PATH_IN_VALIDATE = ""
        self.PATH_IN_TEST = ""

        self.PATH_OUT_LOG = ""
        self.PATH_OUT_MODEL = ""
        self.PATH_OUT_ACCURACY = ""

        self.NETWORK_TYPE = ""
        self.NETWORK_ISTRAINED = False
        self.NETWORK_CLASSCOUNT = 0
        self.ACTIVATION_TYPE = ""

        self.TRANSFORM_SEQUENCE_TRAIN = []
        self.TRANSFORM_SEQUENCE_TRAIN_PARAMETERS = []

        self.TRANSFORM_SEQUENCE_VALIDATE = []
        self.TRANSFORM_SEQUENCE_VALIDATE_PARAMETERS = []

        self.TRANSFORM_SEQUENCE_TEST = []
        self.TRANSFORM_SEQUENCE_TEST_PARAMETERS = []

        # ---- These transformation is for segmentation map
        self.TRANSFORM_SEQUENCE_SEG_END = []
        self.TRANSFORM_SEQUENCE_SEG_END_PARAMETERS = []

        self.LOSS = ""
        self.EPOCH = 0
        self.LEARNING_RATE = 0.001
        self.BATCH = 0

    # ------------------------------------ PUBLIC ------------------------------------

    def getsettings(self):
        """
        Get all the specified settings
        :return: (string) - a string with current settings
        """

        transformTrain = ""
        for i in range(0, len(self.TRANSFORM_SEQUENCE_TRAIN)):
            transformTrain += "[" + self.TRANSFORM_SEQUENCE_TRAIN[i] + "," + \
                                    str(self.TRANSFORM_SEQUENCE_TRAIN_PARAMETERS[i]) + "] "

        transformValidation = ""
        for i in range(0, len(self.TRANSFORM_SEQUENCE_VALIDATE)):
            transformValidation += "[" + self.TRANSFORM_SEQUENCE_VALIDATE[i] + "," + \
                                    str(self.TRANSFORM_SEQUENCE_VALIDATE_PARAMETERS[i]) + "] "

        transformTest = ""
        for i in range(0, len(self.TRANSFORM_SEQUENCE_TEST)):
            transformTest += "[" + self.TRANSFORM_SEQUENCE_TEST[i] + "," + \
                                    str(self.TRANSFORM_SEQUENCE_TEST_PARAMETERS[i]) + "] "

        print(self.TRANSFORM_SEQUENCE_SEG_END)
        print(self.TRANSFORM_SEQUENCE_SEG_END_PARAMETERS)
        transformSegEnd = ""
        for i in range(0, len(self.TRANSFORM_SEQUENCE_SEG_END)):
            transformSegEnd +=  "[" + self.TRANSFORM_SEQUENCE_SEG_END[i] + "," + \
                                    str(self.TRANSFORM_SEQUENCE_SEG_END_PARAMETERS[i]) + "] "

        outputstr = ""

        outputstr += "TASK TYPE: " + self.TASK_TYPE + "\n"
        outputstr += "TASK STAGE: " + self.TASK_STAGE + "\n"
        outputstr += "TASK CHECKPOINT: " + self.TASK_CHECKPOINT + "\n"

        outputstr += "DATABASE: " + self.PATH_IN_ROOT + "\n"
        outputstr += "DATASET TRAIN: " + self.PATH_IN_TRAIN + "\n"
        outputstr += "DATASET VALIDATE: " + self.PATH_IN_VALIDATE + "\n"
        outputstr += "DATASET TEST: " + self.PATH_IN_TEST + "\n"

        outputstr += "OUTPUT LOG: " + self.PATH_OUT_LOG + "\n"
        outputstr += "OUTPUT MODEL: " + self.PATH_OUT_MODEL + "\n"
        outputstr += "OUTPUT ACCURACY: " + self.PATH_OUT_ACCURACY + "\n"
        outputstr += "NETWORK: " + self.NETWORK_TYPE + "\n"
        outputstr += "NETWORK CLASS COUNT: " + str(self.NETWORK_CLASSCOUNT) + "\n"
        outputstr += "NETOWRK PRE-TRAINED: " + str(self.NETWORK_ISTRAINED) + "\n"
        outputstr += "ACTIVATION: " + str(self.ACTIVATION_TYPE) + "\n"
        outputstr += "LOSS: " + self.LOSS + "\n"
        outputstr += "LEARNING RATE: " + str(self.LEARNING_RATE) + "\n"
        outputstr += "TRANSFORM SEQUENCE [TRAIN]: " + transformTrain + "\n"
        outputstr += "TRANSFORM SEQUENCE [VALID]: " + transformValidation + "\n"
        outputstr += "TRANSFORM SEQUENCE [TEST]: " + transformTest + "\n"
        outputstr += "TRANSFROM SEQUENCE [SEG]: " + transformSegEnd
        outputstr += "TRAINING EPOCHS: " + str(self.EPOCH) + "\n"
        outputstr += "BATCH SIZE: " + str(self.BATCH)

        return outputstr

    # --------------------------------------------------------------------------------
    # -------------------------------------- GET -------------------------------------
    # --------------------------------------------------------------------------------

    def getTASK_TYPE(self): return self.TASK_TYPE
    def getTASK_STAGE(self): return self.TASK_STAGE
    def getTASK_CHECKPOINT(self): return self.TASK_CHECKPOINT

    def getPATH_IN_ROOT(self): return self.PATH_IN_ROOT
    def getPATH_IN_TRAIN(self): return self.PATH_IN_TRAIN
    def getPATH_IN_VALIDATE(self): return self.PATH_IN_VALIDATE
    def getPATH_IN_TEST(self): return self.PATH_IN_TEST

    def getPATH_OUT_LOG(self): return self.PATH_OUT_LOG
    def getPATH_OUT_MODEL(self): return self.PATH_OUT_MODEL
    def getPATH_OUT_ACCURACY(self): return self.PATH_OUT_ACCURACY

    def getNETWORK_TYPE(self): return self.NETWORK_TYPE
    def getNETWORK_ISTRAINED(self): return self.NETWORK_ISTRAINED
    def getNETWORK_CLASSCOUNT(self): return self.NETWORK_CLASSCOUNT
    def getACTIVATION_TYPE(self): return  self.ACTIVATION_TYPE

    def getTRANSFORM_SEQUENCE_TRAIN(self): return self.TRANSFORM_SEQUENCE_TRAIN
    def getTRANSFORM_SEQUENCE_TRAIN_ARG(self): return self.TRANSFORM_SEQUENCE_TRAIN_PARAMETERS

    def getTRANSFORM_SEQUENCE_VALIDATE(self): return self.TRANSFORM_SEQUENCE_VALIDATE
    def getTRANSFORM_SEQUENCE_VALIDATE_ARG(self): return self.TRANSFORM_SEQUENCE_VALIDATE_PARAMETERS

    def getTRANSFORM_SEQUENCE_TEST(self): return self.TRANSFORM_SEQUENCE_TEST
    def getTRANSFORM_SEQUENCE_TEST_ARG(self): return self.TRANSFORM_SEQUENCE_TEST_PARAMETERS

    def getTRANSFORM_SEQUENCE_SEG_END(self): return self.TRANSFORM_SEQUENCE_SEG_END
    def getTRANSFORM_SEQUENCE_SEG_END_ARG(self): return self.TRANSFORM_SEQUENCE_SEG_END_PARAMETERS

    def getLOSS(self): return self.LOSS
    def getEPOCH(self): return self.EPOCH
    def getLEARNING_RATE(self): return self.LEARNING_RATE
    def getBATCH(self): return self.BATCH

    # --------------------------------------------------------------------------------
    # -------------------------------------- SET -------------------------------------
    # --------------------------------------------------------------------------------

    def setTASK_TYPE(self, value):
        if not isinstance(value, str): return  False
        if value.upper() not in TaskSettings.TASK_TYPE_LIST: return False
        self.TASK_TYPE = str(value.upper())
        return True

    def setTASK_STAGE(self, value):
        if not isinstance(value, str): return False
        if value.upper() not in TaskSettings.TASK_STAGE_LIST: return False
        self.TASK_STAGE = str(value.upper())
        return True

    def setTASK_CHECKPOINT(self, value):
        self.TASK_CHECKPOINT = str(value)
        return True

    def setPATH_IN_ROOT(self, value):
        self.PATH_IN_ROOT = str(value)
        return True

    def setPATH_IN_TRAIN(self, value):
        self.PATH_IN_TRAIN = str(value)
        return True

    def setPATH_IN_VALIDATE(self, value):
        self.PATH_IN_VALIDATE = str(value)
        return True

    def setPATH_IN_TEST(self, value):
        self.PATH_IN_TEST = str(value)
        return True

    def setPATH_OUT_LOG(self, value):
        self.PATH_OUT_LOG = str(value)
        return True

    def setPATH_OUT_MODEL(self, value):
        self.PATH_OUT_MODEL = str(value)
        return True

    def setPATH_OUT_ACCURACY(self, value):
        self.PATH_OUT_ACCURACY = str(value)
        return True

    def setNETWORK_TYPE(self, value):
        self.NETWORK_TYPE = str(value.upper())
        return True

    def setNETWORK_ISTRAINED(self, value):
        if not isinstance(value, bool): return False
        self.NETWORK_ISTRAINED = value
        return True

    def setNETWORK_CLASSCOUNT(self, value):
        if not isinstance(value, int): return False
        self.NETWORK_CLASSCOUNT = value
        return True

    def setACTIVATION_TYPE(self, value):
        if not isinstance(value, str): return False

        if value.upper() == "SIGMOID":
            self.ACTIVATION_TYPE = 0
            return True

        if value.upper() == "SOFTMAX":
            self.ACTIVATION_TYPE = 1
            return True

        if value.upper() == "" or value.upper() == "NONE":
            self.ACTIVATION_TYPE = 2
            return  True

        return False

    def setTRANSFORM_SEQUENCE_TRAIN(self, value):
        if not isinstance(value, list): return False
        self.TRANSFORM_SEQUENCE_TRAIN = value
        return True

    def setTRANSFORM_SEQUENCE_VALIDATE(self, value):
        if not isinstance(value, list): return False
        self.TRANSFORM_SEQUENCE_VALIDATE = value
        return True

    def setTRANSFORM_SEQUENCE_TEST(self, value):
        if not isinstance(value, list): return False
        self.TRANSFORM_SEQUENCE_TEST = value
        return True

    def setTRANSFORM_SEQUENCE_SEG_END(self, value):
        if not isinstance(value, list): return False
        self.TRANSFORM_SEQUENCE_SEG_END = value
        return True

    def setTRANSFORM_SEQUENCE_TRAIN_ARG(self, value):
        if not isinstance(value, list): return False
        self.TRANSFORM_SEQUENCE_TRAIN_PARAMETERS = value
        return True

    def setTRANSFORM_SEQUENCE_VALIDATE_ARG(self, value):
        if not isinstance(value, list): return False
        self.TRANSFORM_SEQUENCE_VALIDATE_PARAMETERS = value
        return True

    def setTRANSFORM_SEQUENCE_TEST_ARG(self, value):
        if not isinstance(value, list): return False
        self.TRANSFORM_SEQUENCE_TEST_PARAMETERS = value
        return True

    def setTRANSFORM_SEQUENCE_SEG_END_ARG(self, value):
        if not isinstance(value, list): return False
        self.TRANSFORM_SEQUENCE_SEG_END_PARAMETERS=value
        return True

    def setLOSS(self, value):
        self.LOSS = str(value).upper()
        return True

    def setEPOCH(self, value):
        if not isinstance(value, int): return False
        self.EPOCH = value
        return True

    def setLEARNING_RATE(self, value):
        if not isinstance(value, float): return False
        self.LEARNING_RATE = value
        return True

    def setBATCH(self, value):
        if not isinstance(value, int): return False
        self.BATCH = value
        return True
