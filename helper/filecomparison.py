from __future__ import print_function
import numpy as np
import os
import sys
from scipy.spatial import distance
# Please be not to critical. This is a first attempt to compare the files, it is WIP.
# Basically, it is customized to compare the output files from a CUDA and a Python script.
# All input files are expected to contain values which can be read by numpy.loadtxt.
# The script expects a timestamp as a command line argument to find the path of the files.


def comparefiles(pypath, cudaresult, writeresult, dtype):
    # Takes 2 paths for the files to compare and a path to write the result.
    f = open(pypath, 'r')
    ff = open(cudaresult, 'r')
    if f.mode == 'r':
        data = np.loadtxt(f, dtype=dtype, converters={0: lambda s: complex(s.decode().replace('+-', '-').replace('(', '').replace(')',''))})
    if ff.mode == 'r':
        data2 = np.loadtxt(ff, dtype=dtype, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})
    # WIP: other distance measurements might be more meaningful, this is a first try.
    euclideandst = distance.euclidean(data, data2)
    manhattendst = distance.cityblock(data, data2)
    correlationdst = distance.correlation(data, data2)
    # Print the output on cmd.
    print("Euclidiean Distance between the Scripts is:")
    print(euclideandst)
    print("Manhatten Distance between the Scripts is:")
    print(manhattendst)
    print("Correlation between the Scripts is:")
    print(correlationdst)
    # Write the output to custom path.
    result = open(writeresult, "a")
    result.write("Euclidiean Distance:" + str(euclideandst) + "\nManhatten Distance:" + str(manhattendst) + "\nCorrelation:" + str(correlationdst))
    result.close()


if __name__ == '__main__':
    # Paths for the necessary files.
    pathpythonresult = "/mnt/c/Users/b98/PFB_OUTPUT/" + sys.argv[1] + "/Python/output_unformated.txt"
    pathcudaresult = "/mnt/c/Users/b98/PFB_OUTPUT/" + sys.argv[1] + "/output"
    pathfirpythonresult = "/mnt/c/Users/b98/PFB_OUTPUT/" + sys.argv[1] + "/Python/firoutput.txt"
    pathfircudaresult = "/mnt/c/Users/b98/PFB_OUTPUT/" + sys.argv[1] + "/firoutput"
    resultpath = "/mnt/c/Users/b98/PFB_OUTPUT/" + sys.argv[1] + "/result.txt"
    resultpathfir = "/mnt/c/Users/b98/PFB_OUTPUT/" + sys.argv[1] + "/firresult.txt"
    # Check if both files exist and compare intermediate FIR result.
    if os.path.isfile(pathfirpythonresult) and os.path.isfile(pathfircudaresult):
        # TODO: change to int
        comparefiles(pathfirpythonresult, pathfircudaresult, resultpathfir, complex)
    else:
        print("Not all Files necessary for FIR comparison are available.")
    # Check if both files exist and compare endresult.
    if os.path.isfile(pathpythonresult) and os.path.isfile(pathcudaresult):
        comparefiles(pathpythonresult, pathcudaresult, resultpath, complex)
    else:
        print("Not all Files necessary are available.")

