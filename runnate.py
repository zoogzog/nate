import argparse

import nate as nt

# ---------------------- PARSE ARGUMENTS ----------------------
parser = argparse.ArgumentParser()
parser.add_argument("taskfile", help = "path to the task file ")
parser.add_argument("-wc", "--wc", help = "worker count: set 0 for WINDOWS, 8-16 for LINUX based OS", type=int)
parser.add_argument("-c", "--c", help="display console messages", action="store_true")
parser.add_argument("-gpu", "--gpu", help="GPU id to run tests on", type = int)

args = parser.parse_args()

argTaskFile = args.taskfile
argWorkerCount = 0
argSilentMode = True
argGPU = 0

if args.wc is not None: argWorkerCount = args.wc
if args.c: argSilentMode = False
if args.gpu is not None: argGPU = args.gpu

# ---------------------- LAUNCH NATE -------------------------
nt.Nate.run(argTaskFile, argWorkerCount, argSilentMode, argGPU)