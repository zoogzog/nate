import nate as nt

argTaskFile = "./test/task/t-i2i-test.json"
argWorkerCount = 0
argSilentMode = False
argGPU = 0
nt.Nate.run(argTaskFile, argWorkerCount, argSilentMode, argGPU)