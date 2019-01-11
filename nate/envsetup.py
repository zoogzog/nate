import torch
import time
import ntpath

from .datagen import *
from .loss import *
from .network import *
from .settings import *
from .transform import *

# --------------------------------------------------------------------------------
# --- This class is used to initialize environment before carrying out training or
# --- testing tasks.
# --- Author: Andrey G.
# --------------------------------------------------------------------------------

class TaskEnvironment():

    def __init__(self):

        self.NETWORK_MODEL = None
        self.LOSS = None

        self.TRANSFORM_TRAIN = None
        self.TRANSFORM_VALIDATE = None
        self.TRANSFORM_TEST = None
        self.TRANSFORM_SEG = None

        self.DATAGEN_TRAIN = None
        self.DATAGEN_VALIDATE = None
        self.DATAGEN_TEST = None

        self.PATH_IN_ROOT = ""
        self.PATH_IN_TRAIN = ""
        self.PATH_IN_VALIDATE = ""
        self.PATH_IN_TEST = ""

        self.PATH_OUT_LOG = ""
        self.PATH_OUT_MODEL = ""
        self.PATH_OUT_ACCURACY = ""

        self.TIMESTAMP = ""

        self.PATH_FILE_OUT_LOG = ""
        self.PATH_FILE_OUT_MODEL = ""
        self.PATH_FILE_OUT_ACC = ""
        self.PATH_FILE_OUT_PRED = ""

        self.EPOCH = 0
        self.BATCH_SIZE = 0
        self.LR = 0

        self.TEST_TYPE = 0

    def load(self, settings):
        """
        Uses the settings to set-up envrionment before carrying out training/testing procedures
        :param settings: the TaskSettings class object
        :return: 0 or exit code if error occurs
        ER[1]: can't initialize or load the network model
        ER[2]: can't initialize the training transformation
        ER[3]: can't initialize the validation transformation
        ER[4]: can't initialize the testing transformation
        ER[6]: can't initialize the loss function
        ER[7]: can't initialize path environment
        """

        # --- 1. Initialize network model or exit (1) if can't correctly process settings
        isOK = self.initNetworkModel(settings)
        if not isOK: return 1

        # --- 2. Initialize transformations before initializing the data generators
        # --- If can't initialize transformations exit (2, 3, 4, 5)
        isOK = self.initTransformTrain(settings)
        if not isOK: return 2
        isOK = self.initTransformValidate(settings)
        if not isOK: return 3
        isOK = self.initTransformTest(settings)
        if not isOK: return 4
        isOK = self.initTransformSeg(settings)
        if not isOK: return 5

        # --- 3. Initialize datagenerators (transformations should be initialized beforehead), exit (6) if fails
        isOK = self.initDataGenerator(settings)
        if not isOK: return 6

        # --- 4. Initialize loss (do at the end, because sometimes need weights from generators), exit(7) if fails
        isOK = self.initLossFunction(settings)
        if not isOK: return 7

        # --- 5. Initialize other environment variables, exit(8) if fails
        isOK = self.initEnvironment(settings)
        if not isOK: return 8

        return 0


    # --------------------------------------------------------------------------------
    # -------------------------------------- INIT ------------------------------------
    # --------------------------------------------------------------------------------

    def initNetworkModel(self, settings):

        taskStage = settings.getTASK_STAGE()

        networkName = settings.getNETWORK_TYPE()
        networkClassCount = settings.getNETWORK_CLASSCOUNT()
        networkActivation = settings.getACTIVATION_TYPE()
        networkIsTrained = settings.getNETWORK_ISTRAINED()

        # ---- Generate the network using network factory, exit if can't do it
        self.NETWORK_MODEL = NetworkFactory.getNetwork(networkName, networkClassCount, networkActivation, networkIsTrained)
        if self.NETWORK_MODEL == None: return False

        # ---- Change to parallel mode
        self.NETWORK_MODEL = torch.nn.DataParallel(self.NETWORK_MODEL).cuda()

        # ---- If we want to do testing or resume training then load the network from checkpoint
        if taskStage == TaskSettings.TASK_STAGE_RESUME or taskStage == TaskSettings.TASK_STAGE_TEST:
            try:
                checkpointPath = settings.getTASK_CHECKPOINT()
                checkpointModel = torch.load(checkpointPath)
                self.NETWORK_MODEL.load_state_dict(checkpointModel['state_dict'])
            except:
                return False

        return True

    # --------------------------------------------------------------------------------

    def initDataGenerator(self, settings):

        taskStage = settings.getTASK_STAGE()

        # ---- Initialize generators for training stage - TRAIN, VALIDATE
        if taskStage == TaskSettings.TASK_STAGE_TRAIN:
            if not self.initDataGeneratorTrain(settings): return False
            if not self.initDataGeneratorValidate(settings): return False

        # ---- Initialize generators for testing stage - TEST
        if taskStage == TaskSettings.TASK_STAGE_TEST:
            if not self.initDataGeneratorTest(settings): return False

        # ---- Initialize generators for resuming stage (e2e resume) - TRAIN, VALIDATE, TEST
        if taskStage == TaskSettings.TASK_STAGE_RESUME:
            if not self.initDataGeneratorTrain(settings): return False
            if not self.initDataGeneratorValidate(settings): return False
            if not self.initDataGeneratorTest(settings): return False

        # ---- Initialize generators for end to end workflow (e2e) - TRAIN, VALIDATE, TEST
        if taskStage == TaskSettings.TASK_STAGE_S2E:
            if not self.initDataGeneratorTrain(settings): return False
            if not self.initDataGeneratorValidate(settings): return False
            if not self.initDataGeneratorTest(settings): return False

        return True

    def initDataGeneratorTrain(self, settings):

        pathInRoot = settings.getPATH_IN_ROOT()
        pathInTrain = settings.getPATH_IN_TRAIN()

        taskType = settings.getTASK_TYPE()
        generatorType = ""

        # ---- Select data generator depending on the type of task
        if taskType == TaskSettings.TASK_TYPE_CLASS:
            generatorType = DatagenFactory.DATAGEN_CLASSIFICATION
            self.DATAGEN_TRAIN = DatagenFactory.getDatagen(generatorType, pathInRoot, pathInTrain, [self.TRANSFORM_TRAIN, self.TRANSFORM_TRAIN])
        if taskType == TaskSettings.TASK_TYPE_IMG2IMG:
            generatorType = DatagenFactory.DATAGEN_AUTOENCODER
            self.DATAGEN_TRAIN = DatagenFactory.getDatagen(generatorType, pathInRoot, pathInTrain, [self.TRANSFORM_TRAIN, self.TRANSFORM_TRAIN])
        if taskType == TaskSettings.TASK_TYPE_SEG:
            generatorType = DatagenFactory.DATAGEN_SEGMENTATION
            self.DATAGEN_TRAIN = DatagenFactory.getDatagen(generatorType, pathInRoot, pathInTrain, [self.TRANSFORM_TRAIN, self.TRANSFORM_SEG])

        if self.DATAGEN_TRAIN is None: return False

        return True

    def initDataGeneratorValidate(self, settings):

        pathInRoot = settings.getPATH_IN_ROOT()
        pathInValidate = settings.getPATH_IN_VALIDATE()

        taskType = settings.getTASK_TYPE()
        generatorType = ""

        # ---- Select data generator depending on the type of task
        if taskType == TaskSettings.TASK_TYPE_CLASS:
            generatorType = DatagenFactory.DATAGEN_CLASSIFICATION
            self.DATAGEN_VALIDATE = DatagenFactory.getDatagen(generatorType, pathInRoot, pathInValidate, [self.TRANSFORM_VALIDATE, self.TRANSFORM_VALIDATE])
        if taskType == TaskSettings.TASK_TYPE_IMG2IMG:
            generatorType = DatagenFactory.DATAGEN_AUTOENCODER
            self.DATAGEN_VALIDATE = DatagenFactory.getDatagen(generatorType, pathInRoot, pathInValidate, [self.TRANSFORM_VALIDATE, self.TRANSFORM_VALIDATE])
        if taskType == TaskSettings.TASK_TYPE_SEG:
            generatorType = DatagenFactory.DATAGEN_SEGMENTATION
            self.DATAGEN_VALIDATE = DatagenFactory.getDatagen(generatorType, pathInRoot, pathInValidate, [self.TRANSFORM_VALIDATE, self.TRANSFORM_SEG])

        if self.DATAGEN_VALIDATE is None: return False

        return True

    def initDataGeneratorTest(self, settings):

        pathInRoot = settings.getPATH_IN_ROOT()
        pathInTest = settings.getPATH_IN_TEST()

        taskType = settings.getTASK_TYPE()
        generatorType = ""

        # ---- Select data generator depending on the type of task
        if taskType == TaskSettings.TASK_TYPE_CLASS:
            generatorType = DatagenFactory.DATAGEN_CLASSIFICATION
            self.DATAGEN_TEST = DatagenFactory.getDatagen(generatorType, pathInRoot, pathInTest, [self.TRANSFORM_TEST, self.TRANSFORM_TEST])
        if taskType == TaskSettings.TASK_TYPE_IMG2IMG:
            generatorType = DatagenFactory.DATAGEN_AUTOENCODER
            self.DATAGEN_TEST = DatagenFactory.getDatagen(generatorType, pathInRoot, pathInTest, [self.TRANSFORM_TEST, self.TRANSFORM_TEST])
        if taskType == TaskSettings.TASK_TYPE_SEG:
            generatorType = DatagenFactory.DATAGEN_SEGMENTATION
            self.DATAGEN_TEST = DatagenFactory.getDatagen(generatorType, pathInRoot, pathInTest, [self.TRANSFORM_TEST, self.TRANSFORM_SEG])

        if self.DATAGEN_TEST is None: return False

        return True

    # --------------------------------------------------------------------------------

    def initTransformTrain(self, settings):
        # ---- If this is a test-only task, then no need initializing training trasnformation
        taskStage = settings.getTASK_STAGE()
        if taskStage == TaskSettings.TASK_STAGE_TEST: return True

        transformTrain = settings.getTRANSFORM_SEQUENCE_TRAIN()
        transformTrainArgs = settings.getTRANSFORM_SEQUENCE_TRAIN_ARG()

        self.TRANSFORM_TRAIN = TransformFactory.getTransformSequence(transformTrain, transformTrainArgs, True)
        if self.TRANSFORM_TRAIN is None: return False

        return True

    def initTransformValidate(self, settings):
        # ---- If this is a test-only task, then no need initializing validation trasnformation
        taskStage = settings.getTASK_STAGE()
        if taskStage == TaskSettings.TASK_STAGE_TEST: return True

        transformValidate = settings.getTRANSFORM_SEQUENCE_VALIDATE()
        transformValidateArgs = settings.getTRANSFORM_SEQUENCE_VALIDATE_ARG()

        self.TRANSFORM_VALIDATE = TransformFactory.getTransformSequence(transformValidate, transformValidateArgs, True)
        if self.TRANSFORM_VALIDATE is None: return False

        return True

    def initTransformTest(self, settings):
        # ---- If this is train-only task, then no need initializing training transformation
        taskStage = settings.getTASK_STAGE()
        if taskStage == TaskSettings.TASK_STAGE_TRAIN: return True

        transformTest = settings.getTRANSFORM_SEQUENCE_TEST()
        transformTestArgs = settings.getTRANSFORM_SEQUENCE_TEST_ARG()

        self.TRANSFORM_TEST = TransformFactory.getTransformSequence(transformTest, transformTestArgs, True)
        if self.TRANSFORM_TEST is None: return False

        return True

    def initTransformSeg(self, settings):
        transformSeg = settings.getTRANSFORM_SEQUENCE_SEG_END()
        transformSegArgs = settings.getTRANSFORM_SEQUENCE_SEG_END_ARG()

        self.TRANSFORM_SEG = TransformFactory.getTransformSequenceMap(transformSeg, transformSegArgs)

        if self.TRANSFORM_SEG is None: return False
        return True

    # --------------------------------------------------------------------------------

    def initLossFunction(self, settings):
        # ---- We don't need loss if we want to conduct only testing
        taskStage = settings.getTASK_STAGE()
        if taskStage == TaskSettings.TASK_STAGE_TEST: return True

        lossName = settings.getLOSS()
        self.LOSS = LossFactory.getLossFunction(lossName, self.DATAGEN_TRAIN. getfrequency())
        if self.LOSS is None: return False

        return True

    def initEnvironment(self, settings):
        self.PATH_IN_ROOT = settings.getPATH_IN_ROOT()
        self.PATH_IN_TRAIN = settings.getPATH_IN_TRAIN()
        self.PATH_IN_VALIDATE = settings.getPATH_IN_VALIDATE()
        self.PATH_IN_TEST = settings.getPATH_IN_TEST()

        self.PATH_OUT_LOG = settings.getPATH_OUT_LOG()
        self.PATH_OUT_MODEL = settings.getPATH_OUT_MODEL()
        self.PATH_OUT_ACCURACY = settings.getPATH_OUT_ACCURACY()

        taskStage = settings.getTASK_STAGE()

        # ---- Generate a new timestamp for TRAIN and E2E stages
        # ---- Timestamp used as the ID of the experiment
        if taskStage == TaskSettings.TASK_STAGE_TRAIN or taskStage == TaskSettings.TASK_STAGE_S2E:
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampStart = timestampDate + '-' + timestampTime
            self.TIMESTAMP = timestampStart

        # ---- If the stage is RESUME or TEST initialize the timestamp from the model checkpoint
        if taskStage == TaskSettings.TASK_STAGE_TEST or taskStage == TaskSettings.TASK_STAGE_RESUME:
            taskCheckpoint = settings.getTASK_CHECKPOINT()

            # ---- Check if the checkpoint is of the correct format
            # ---- Most likely at this point we have already got an exception let's be extra careful here
            if ".pth.tar" not in taskCheckpoint: return False

            checkpointName = os.path.basename(taskCheckpoint)
            checkpointName = checkpointName[2:-8]

            self.TIMESTAMP = checkpointName

        # ---- Generate file paths
        fileModel = "m-" + self.TIMESTAMP+ ".pth.tar"
        fileLog = "log-" + self.TIMESTAMP + ".txt"
        fileAcc = "acc-" + self.TIMESTAMP + ".txt"
        filePred = "pred-" + self.TIMESTAMP + ".txt"

        self.PATH_FILE_OUT_MODEL = os.path.join( self.PATH_OUT_MODEL, fileModel)
        self.PATH_FILE_OUT_LOG = os.path.join(self.PATH_OUT_LOG, fileLog)
        self.PATH_FILE_OUT_ACC = os.path.join(self.PATH_OUT_ACCURACY, fileAcc)
        self.PATH_FILE_OUT_PRED = os.path.join(self.PATH_OUT_ACCURACY, filePred)

        self.EPOCH = settings.getEPOCH()
        self.BATCH_SIZE = settings.getBATCH()
        self.LR = settings.getLEARNING_RATE()

        #---- Select the testing procedure
        if TransformFactory.TRANSFORM_10CROP.lower() in settings.getTRANSFORM_SEQUENCE_TEST(): self.TEST_TYPE = 1
        if TransformFactory.TRANSFORM_10CROP in settings.getTRANSFORM_SEQUENCE_TEST(): self.TEST_TYPE = 1
        if settings.getTASK_TYPE() == TaskSettings.TASK_TYPE_IMG2IMG: self.TEST_TYPE = 2
        if settings.getTASK_TYPE() == TaskSettings.TASK_TYPE_SEG: self.TEST_TYPE = 3

        return True

    # --------------------------------------------------------------------------------
    # -------------------------------------- GET -------------------------------------
    # --------------------------------------------------------------------------------

    def getNETWORK_MODEL(self): return self.NETWORK_MODEL
    def getLOSS(self): return self.LOSS
    def getDATAGEN_TRAIN(self): return self.DATAGEN_TRAIN
    def getDATAGEN_VALIDATE(self): return self.DATAGEN_VALIDATE
    def getDATAGEN_TEST(self): return self.DATAGEN_TEST

    def getPATH_FILE_OUT_LOG(self): return  self.PATH_FILE_OUT_LOG
    def getPATH_FILE_OUT_MODEL(self): return self.PATH_FILE_OUT_MODEL
    def getPATH_FILE_OUT_ACC(self): return self.PATH_FILE_OUT_ACC
    def getPATH_FILE_OUT_PRED(self): return self.PATH_FILE_OUT_PRED

    def getPATH_OUT_ACC(self): return self.PATH_OUT_ACCURACY

    def getEPOCH(self): return self.EPOCH
    def getBATCH_SIZE(self): return self.BATCH_SIZE
    def getLEARNING_RATE(self): return self.LR

    def getTEST_TYPE(self): return self.TEST_TYPE