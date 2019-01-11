import json
import sys

from .settings import *

class TaskSettingsCollection():

    # --------------------------------------------------------------------------------
    TASK_LIST = "tasklist"

    TASK_TYPE = "tasktype"
    TASK_STAGE = "taskstage"
    TASK_CHECKPOINT = "taskcheckpoint"

    TASK_DATABASE = "database"
    TASK_DATABASE_TRAIN = "dataset_train"
    TASK_DATABASE_VALIDATE = "dataset_validate"
    TASK_DATABASE_TEST = "dataset_test"

    TASK_OUTPUT_LOG = "output_log"
    TASK_OUTPUT_MODEL = "output_model"
    TASK_OUTPUT_ACCURACY = "output_accuracy"

    TASK_NN_MODEL = "network"
    TASK_NN_ISTRAINED = "network_istrained"
    TASK_NN_CLASSCOUNT = "network_classcount"
    TASK_NN_ACTIVATION = "activation"

    TASK_TRNSFRM_TRAIN = "trnsfrm_train"
    TASK_TRNSFRM_TRAIN_PARAM = "trnsfrm_train_param"

    TASK_TRNSFRM_VALIDATE = "trnsfrm_validate"
    TASK_TRNSFRM_VALIDATE_PARAM = "trnsfrm_validate_param"

    TASK_TRNSFRM_TEST = "trnsfrm_test"
    TASK_TRNSFRM_TEST_PARAM = "trnsfrm_test_param"

    TASK_TRNSFRM_SEG = "trnsfrm_seg_end"
    TASK_TRNSFRM_SEG_PARAM = "trnsfrm_seg_end_param"

    TASK_LOSS = "loss"
    TASK_EPOCH = "epoch"
    TASK_LRATE = "lrate"
    TASK_BATCH = "batch"
    # --------------------------------------------------------------------------------

    def __init__(self):
        self.taskcollection = []
        self.taskcount = 0

    # --------------------------------------------------------------------------------

    def load(self, path):
        """
        Load tasks from a JSON file
        :param path: path to the JSON file that contains task settings data
        """
        with open(path) as f:
            data = json.load(f)

        self.taskcount = len(data[TaskSettingsCollection.TASK_LIST])

        for i in range(0, self.taskcount):

            task = TaskSettings()

            taskdata = data[TaskSettingsCollection.TASK_LIST][i]

            for key, value in taskdata.items():

                # ---- TASK TYPE, STAGE, CHECKPOINT
                if key == TaskSettingsCollection.TASK_TYPE:
                    if not task.setTASK_TYPE(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter TASK_TYPE!")
                        return False
                if key == TaskSettingsCollection.TASK_STAGE:
                    if not task.setTASK_STAGE(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter TASK_STAGE!")
                        return False
                if key == TaskSettingsCollection.TASK_CHECKPOINT:
                    if not task.setTASK_CHECKPOINT(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter TASK_CHECKPOINT")
                        return False

                # ----- DATABASE and DATASETS, (other environment variables)
                if key == TaskSettingsCollection.TASK_DATABASE:
                    if not task.setPATH_IN_ROOT(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter PATH_IN_ROOT!")
                        return False
                if key == TaskSettingsCollection.TASK_DATABASE_TRAIN:
                    if not task.setPATH_IN_TRAIN(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter PATH_IN_TRAIN!")
                        return False
                if key == TaskSettingsCollection.TASK_DATABASE_VALIDATE:
                    if not task.setPATH_IN_VALIDATE(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter PATH_IN_VALIDATE!")
                        return False
                if key == TaskSettingsCollection.TASK_DATABASE_TEST:
                    if not task.setPATH_IN_TEST(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter PATH_IN_TEST!")
                        return False

                if key == TaskSettingsCollection.TASK_OUTPUT_ACCURACY:
                    if not task.setPATH_OUT_ACCURACY(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter PATH_OUT_ACCURACY!")
                        return False
                if key == TaskSettingsCollection.TASK_OUTPUT_LOG:
                    if not task.setPATH_OUT_LOG(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter PATH_OUT_LOG!")
                        return False
                if key == TaskSettingsCollection.TASK_OUTPUT_MODEL:
                    if not task.setPATH_OUT_MODEL(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter PATH_OUT_MODEL!")
                        return False

                # ----- NETWORK SETTINGS
                if key == TaskSettingsCollection.TASK_NN_ACTIVATION:
                    if not task.setACTIVATION_TYPE(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter ACTIVATION_TYPE!")
                        return False
                if key == TaskSettingsCollection.TASK_NN_CLASSCOUNT:
                    if not task.setNETWORK_CLASSCOUNT(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter NETWORK_CLASS_COUNT!")
                        return False
                if key == TaskSettingsCollection.TASK_NN_ISTRAINED:
                    if not task.setNETWORK_ISTRAINED(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter !")
                        return False
                if key == TaskSettingsCollection.TASK_NN_MODEL:
                    if not task.setNETWORK_TYPE(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter NETWORK_TYPE!")
                        return False

                # ----- TRANSFORMATIONS
                if key == TaskSettingsCollection.TASK_TRNSFRM_TRAIN:
                    if not task.setTRANSFORM_SEQUENCE_TRAIN(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter TRANSFORM_SEQ_TRAIN!")
                        return False
                if key == TaskSettingsCollection.TASK_TRNSFRM_TRAIN_PARAM:
                    if not task.setTRANSFORM_SEQUENCE_TRAIN_ARG(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter TRANSFORM_SEQ_TRAIN_ARG!")
                        return False
                if key == TaskSettingsCollection.TASK_TRNSFRM_VALIDATE:
                    if not task.setTRANSFORM_SEQUENCE_VALIDATE(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter TRANSFORM_SEQ_VALIDATE!")
                        return False
                if key == TaskSettingsCollection.TASK_TRNSFRM_VALIDATE_PARAM:
                    if not task.setTRANSFORM_SEQUENCE_VALIDATE_ARG(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter TRANSFORM_SEQ_VALIDATE_ARG!")
                        return False
                if key == TaskSettingsCollection.TASK_TRNSFRM_TEST:
                    if not task.setTRANSFORM_SEQUENCE_TEST(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter TRANSFORM_SEQ_TEST!")
                        return False
                if key == TaskSettingsCollection.TASK_TRNSFRM_TEST_PARAM:
                    if not task.setTRANSFORM_SEQUENCE_TEST_ARG(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter TRANSFORM_SEQ_TEST_ARG!")
                        return False
                if key == TaskSettingsCollection.TASK_TRNSFRM_SEG:
                    if not task.setTRANSFORM_SEQUENCE_SEG_END(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter TRANSFORM_SEQ_SEG_END!")
                        return  False
                if key == TaskSettingsCollection.TASK_TRNSFRM_SEG_PARAM:
                    if not task.setTRANSFORM_SEQUENCE_SEG_END_ARG(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter TRANSFORM_SEQ_SEG_EDN_ARG!")
                        return False

                # ---- Other parameters - loss, epoch, lrate, batch
                if key == TaskSettingsCollection.TASK_LOSS:
                    if not task.setLOSS(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter LOSS!")
                        return  False
                if key == TaskSettingsCollection.TASK_EPOCH:
                    if not task.setEPOCH(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter EPOCH!")
                        return False
                if key == TaskSettingsCollection.TASK_LRATE:
                    if not task.setLEARNING_RATE(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter LEARNING RATE!")
                        return False
                if key == TaskSettingsCollection.TASK_BATCH:
                    if not task.setBATCH(value):
                        print("Error TASK_ID=" + str(i) + ": wrong parameter BATCH!")
                        return False


            self.taskcollection.append(task)

        return True

    def gettask(self, index):
        """
        Get task by its index
        :param index: index of the desired task
        :return: (TaskSettings) - task settings
        """

        if(index >= 0) and (index < self.taskcount):
            return self.taskcollection[index]

        return None

    def getsize(self):
        return len(self.taskcollection)

