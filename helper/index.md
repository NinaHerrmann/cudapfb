# Helper Files

**Attention: this will only work for Windows Systems with VS!**

During the project, the accuracy of the results has to be ensured. However, this is a challenging process when it comes to big testing data. This is particularly the case when the size of the data makes it time-consuming and error-prone to manually calculate the results. 

This problem is tackled by comparing the results from the CUDA script to results from a python script. The given helper files in this repository serve this purpose. However do not overstrain your PC! Keep in mind that all calculation are done twice and that the Python script is not executed in parallel!

## Preconditions

Before running the Run.bat script which subsumes the comparing process a few steps have to be taken.

1. The File assumes that Visual Studio 17 is used. Therefore, MSBuild is expected to be on the following path: `C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin\MSBuild.exe` Please check if it is available on your system under the given path and change it otherwise

2. The CUDA program is expected to be available under  `C:\PFB\low-level\PFB_radioastronomie\PFB_radioastronomie.sln` Please clone the repository into `C:` and call it PFB, or change the path in the `Run.bat`

3. The CUDA program writes files to the path `C:\Users\b98\PFB_OUTPUT`. I know this is a major shortcoming and you will have to replace it with your own username. However, there exists simple replace scripts which should serve this purpose ([see e.g. here](https://stackoverflow.com/questions/41542554/batch-script-to-replace-specific-string-in-multiple-files)).

4. During the development phase, some problems occurred with python. The script launches the Ubuntu App which is available for Windows and executes all python code there.  It expects to have the two helper files `filecomparison.py` and `pfb.py` in the `root` directory of the Ubuntu App. However, if you prefer to use Python on Windows, remove the command `wsl` from the `Run.bat` and adjust the `root` path to the path where you store the `.py` files, most likely `C:\PFB\helper\filecomparison.py` and  `C:\PFB\helper\pfb.py` . Please also make sure that you have installed the Python packages: numpy, os, sys, scipy, (scipy.spatial),(scipy.signal), and matplotlib. 

   

   The results will be written to `C:\Users\b98\PFB_OUTPUT\__timestamp__\`, where the timestamp is the time the CUDA program is executed. It is generated as part of the C script. Don't get worried about the other files, if you only care about the results, `result.txt` and `firresult.txt` give you the Euclidean distance, the Manhattan distance, and the correlation between the results of the scripts. `firresult.txt` compares the results of the FIR Filter, and results compares the output of the Combination of FIR-Filter and the application of the Fast Fourier Transform.

   ## Run the Script

   To run the script, start Run.bat in the prompt with the number of taps, the number of channels, and the number of spectra as arguments.

   ## Understanding the Script and Intermediate Results

   This script is perfectly fine when the programs work correctly. However, in case of differences, it is helpful to have a look at the intermediate results.
   
   First, it is helpful to check the `firresult.txt`. If the FIR-Filter has different outcomes the discrete Fourier Transformation is executed on varying data and therefore, extremely unlikely to be equal. If the similarity measures are significant the single FIR result files can be checked (`C:\Users\b98\PFB_OUTPUT\__timestamp__\firoutput` and `C:\Users\b98\PFB_OUTPUT\__timestamp__\Python\firoutput`). Are all numbers different or is there a reoccurring pattern? Please consider that the Python file which produces the `Python\firoutput` file is based on the same input data so most likely the CUDA script produces the wrong results or the data is imported wrong. 
   
   Second, the results of the FIR filter are alike, but the DFT results have significant differences. Again it can be checked whether all results are different or only single results are different by comparing the two output files (`C:\Users\b98\PFB_OUTPUT\__timestamp__\output` and `C:\Users\b98\PFB_OUTPUT\__timestamp__\Python\output_unformatted`). Debugging becomes a hard task here, for finding the program needs to be precisely scanned. However, keep in mind, that the Python script is an official version and most likely correct, while the CUDA script is optimized and therefore more error-prone.
   
   ## WIP and Future Features
   
   - [x] Pass #Channels, #Taps, and #Spectra as optional Arguments instead of setting them manually in the CUDA script.
   - [ ] Add the gcc compiler.