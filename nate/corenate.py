from .workflow import *
import os


class Nate():

    def run(taskJSON, workerCount = 0, isSilentMode = True, GPU_ID = 0):
        """
        The main entry into the Network Auto Tester and Evaluator (NATE)
        :param taskJSON: path to the JSON file with the description of tasks
        :param workerCount: number of workers (0 for WIN, 8-16 for base LINUX)
        :param isSilentMode: flag which is True for totally silent mode, False - all messages displayed
        :param GPU_ID: the ID of the GPU to run tasks on (NOT USED IN V.3)
        :return:
        """

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

        # ---- Parse the JSON task
        taskList = TaskSettingsCollection()
        isOK = taskList.load(taskJSON)

        # ----- EXCEPTION: can't load JSON file correctly
        if not isOK:
            print("EXCEPTION: Can't parse JSON task file correctly!")

        # ---- Go over tasks and do execution
        for i in range (0, taskList.getsize()):
            task = taskList.gettask(i)

            if not isSilentMode: print ("Launching task ID=" + str(i))

            try:
                TaskLauncher.launch(task, workerCount, isSilentMode)
            except Exception as e:
                if not isSilentMode:
                    print("EXCEPTION: Run-time exception occurred during task ID=" + str(i))
                    print("---------------------------------------------------------------")
                    print(e)
                    print("---------------------------------------------------------------")
                    print("EXCEPTION: Switching to the next task in the list")







