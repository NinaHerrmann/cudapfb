import numpy as np
import os
import sys


def comparefiles(pythonresult, cudaresult):
    f = open(pythonresult, 'r')
    ff = open(cudaresult, 'r')
    if f.mode == "r":
        data = np.loadtxt(f)
        print(data)
    if ff.mode == "r":
        data2 = np.loadtxt(ff)
        print(data2)
    print("Successfully finished comparefiles.")
    return True


if __name__ == '__main__':
    pathpythonresult = "/mnt/c/Users/b98/PFB_OUTPUT/" + sys.argv[1] + "/Python/output_unformated"
    pathcudaresult = "/mnt/c/Users/b98/PFB_OUTPUT/" + sys.argv[1] + "/f_output"
    if os.path.isfile(pathpythonresult) and os.path.isfile(pathcudaresult):
        comparefiles(pathpythonresult, pathcudaresult)
    else:
        print("Not all Files necessary are available.")
