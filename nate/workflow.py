import time
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .corealg import *
from .settings import *
from .envsetup import *

class TaskLauncher():

    def train(environment, workerCount = 0, isSilentMode = False):

        if not isSilentMode: print("TRAINING: Initializing datasets")
        datasetTrain = environment.getDATAGEN_TRAIN()
        datasetValidation = environment.getDATAGEN_VALIDATE()

        if not isSilentMode: print("TRAINING: Initializing data loaders wk=" + str(workerCount))
        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=environment.getBATCH_SIZE(), shuffle=True, num_workers=workerCount, pin_memory=False)
        dataLoaderValidation = DataLoader(dataset=datasetValidation, batch_size=environment.getBATCH_SIZE(), shuffle=False, num_workers=workerCount, pin_memory=False)

        if not isSilentMode: print("TRAINING: Initializing optimizer and scheduler")
        optimizer = optim.Adam(environment.getNETWORK_MODEL().parameters(), lr=environment.getLEARNING_RATE(), betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10, mode='min')

        if not isSilentMode: print("TRAINING: Launching training loop")
        CoreAlgorithm.train(
            dataLoaderTrain,
            dataLoaderValidation,
            environment.getNETWORK_MODEL(),
            environment.getEPOCH(),
            environment.getLOSS(),
            optimizer,
            scheduler,
            environment.getPATH_FILE_OUT_MODEL(),
            environment.getPATH_FILE_OUT_LOG(),
            isSilentMode)

    def test(environment, workerCount=0, isSilentMode=False):

        if not isSilentMode: print("TESTING: Initializing dataset for testing")
        datasetTest = environment.getDATAGEN_TEST()

        if not isSilentMode: print("TESTING: Initializing data loader wk=" + str(workerCount))
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=environment.getBATCH_SIZE(), shuffle=False, num_workers=workerCount, pin_memory=False)

        if not isSilentMode: print("TESTING: Launching testing loop")


        testType = environment.getTEST_TYPE()

        # ---- Testing for the classification task without 10 crop transformation
        if testType == 0: CoreAlgorithm.test(dataLoaderTest, environment.getNETWORK_MODEL(), environment.getPATH_FILE_OUT_ACC(), environment.getPATH_FILE_OUT_PRED(), isSilentMode)
        # ---- Testing for the classification task with 10 crop transformation
        if testType == 1: CoreAlgorithm.test10crop(dataLoaderTest, environment.getNETWORK_MODEL(), environment.getPATH_FILE_OUT_ACC(), environment.getPATH_FILE_OUT_PRED(), isSilentMode)
        # ---- Testing for image to image task (autoencoder)
        if testType == 2: CoreAlgorithm.testi2i(dataLoaderTest, environment.getNETWORK_MODEL(), environment.getPATH_OUT_ACC(), isSilentMode)
        if testType == 3: CoreAlgorithm.testseg(dataLoaderTest, datasetTest, environment.getNETWORK_MODEL(), environment.getPATH_OUT_ACC(), isSilentMode)



    def launch(settings, workerCount = 0, isSilentMode = False):
        """
        Launch a single task
        :param settings: TaskSettings class
        """

        taskStage = settings.getTASK_STAGE()
        taskType = settings.getTASK_TYPE()

        environment = TaskEnvironment()
        exitCode = environment.load(settings)

        # ---- EXCEPTION!!!
        if exitCode != 0:
            if not isSilentMode:
                print("EXCEPTION: Environment setup is failed with code " + str(exitCode))
                return exitCode

        # --------------------------------------------------------------------------------------------------
        # ---- Train the network if the task is TRAIN, RESUME, E2E
        if taskStage == TaskSettings.TASK_STAGE_TRAIN or taskStage == TaskSettings.TASK_STAGE_RESUME or taskStage == TaskSettings.TASK_STAGE_S2E:

            # ---- LOG: display in console if not in silend mode
            if not isSilentMode:
                print("--------------------------------------------------")
                print(settings.getsettings())
                print("--------------------------------------------------")
                print("[TASK-CLASSIFICATION] Launching training procedure")

            # ---- LOG: write info to the log file
            ostreamLog = open(environment.getPATH_FILE_OUT_LOG(), 'w')
            ostreamLog.write(settings.getsettings())
            ostreamLog.close()

            # <<<<<<<<<<<<<<<<<<<<<<< TRAIN >>>>>>>>>>>>>>>>>>>>>>
            TaskLauncher.train(environment, workerCount, isSilentMode)
            # <<<<<<<<<<<<<<<<<<<<<<< TRAIN >>>>>>>>>>>>>>>>>>>>>>

        # --------------------------------------------------------------------------------------------------
        # ---- Test the network if the task is TEST, RESUME, E2E
        if taskStage == TaskSettings.TASK_STAGE_TEST or taskStage == TaskSettings.TASK_STAGE_RESUME or taskStage == TaskSettings.TASK_STAGE_S2E:

            # ---- LOG: display in console if not in silent mode
            if not isSilentMode:
                print("--------------------------------------------------")
                print("[TASK-CLASSIFICATION] Launching testing procedure")

            # <<<<<<<<<<<<<<<<<<<<<<< TEST >>>>>>>>>>>>>>>>>>>>>>>
            TaskLauncher.test(environment, workerCount, isSilentMode)
            # <<<<<<<<<<<<<<<<<<<<<<< TEST >>>>>>>>>>>>>>>>>>>>>>>


        return 0