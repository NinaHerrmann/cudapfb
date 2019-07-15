:: WIP!
:: 0.0 Get Variables from CL
set nTap=%1
set nChan=%2
set nSpectra=%3
:: 0.1 Write Arguments to params file so they can be const and global
SET MYFILE="C:\PFB\low-level\PFB_radioastronomie\PFB_radioastronomie\params.h"
IF EXIST %MYFILE% DEL /F %MYFILE%
(
  echo #pragma once
  echo #ifndef TAPS
  echo #define TAPS %nTap%
  echo #endif
  echo // Number of Channels:
  echo #ifndef CHANNELS
  echo #define CHANNELS %nChan%
  echo #endif
  echo // Number of Spectra:
  echo #ifndef NSPECTRA
  echo #define NSPECTRA %nSpectra%
  echo #endif
  echo // Configuarable Settings which change e.g. memory size and increase or slow down computations dependent on the architecture and computation size:
  echo // From the original github https://github.com/AstroAccelerateOrg/astro-accelerate/tree/master/lib/AstroAccelerate/PPF:
  echo // Warning: Calculation of these parameters is different for each precision case. Sorry.
  echo // This is a dummy parameter, which is not used in the code itself. It says how many thread-blocks we would like to be resident on single SM 
  echo #ifndef ACTIVE_BLOCKS
  echo #define ACTIVE_BLOCKS 3
  echo #endif
  echo // This is again dummy parameter. It says how many single precision floating point numbers can fit into shared memory available to single block
  echo #ifndef TOT_SM_SIZE
  echo #define TOT_SM_SIZE 12288
  echo #endif
  echo // This is again dummy parameter which says how many channels are processed per single thread-block. It is accualy size of a warp=32.
  echo #ifndef CHANNELS_PER_BLOCK
  echo #define CHANNELS_PER_BLOCK 32
  echo #endif
  echo // note: for Maxwell generation the ACTIVE_BLOCKS could be half of blocks present on single SM as this generation has 96kB of shared memory, but shared memory per block is still 48kB
  echo // WARP size
  echo // Remark: In case you wonder why I cannot spell WARP, Actually, the thrust <thrust/host_vector.h> and <thrust/device_vector.h> libraries also have a global variable WARP making compilation impossible when it is called the same name.
  echo #ifndef WARPP
  echo #define WARPP 32
  echo #endif
  echo // COEFF_SIZE is given by number of taps and channels processed per thread-block COEFF_SIZE=CHANNELS_PER_BLOCK*pfb_ntaps=512
  echo // TODO: It is always allocalted as shared memory. we might want to change this if there are to much coeff to be in shared memory.
  echo #ifndef COEFF_SIZE
  echo #define COEFF_SIZE 512
  echo #endif
  echo // DATA_SIZE says how many input data elements in floating point numbers we want to store in shared memory per thread-block. DATA_SIZE=TOT_SM_SIZE/ACTIVE_BLOCKS=4096; DATA_SIZE=DATA_SIZE-COEFF_SIZE=3584; this is because we need to store coefficients in the shared memory along with the input data. Lastly we must divide this by two DATA_SIZE=DATA_SIZE/2=1792; This is because we store real and imaginary input data separately to prevent bank conflicts. 
  echo #ifndef DATA_SIZE
  echo #define DATA_SIZE 1792
  echo #endif
  echo // THREADS_PER_BLOCK gives number of threads per thread-block. It could be calculated as such THREADS_PER_BLOCK=MAX_THREADS_PER_SM/ACTIVE_BLOCKS; rounded to nearest lower multiple of 32; In case of Maxwell generation MAX_THREADS_PER_SM=2048, thus THREADS_PER_BLOCK=682.6666, rounding to nearest lower multiple of 32 gives THREADS_PER_BLOCK=672;
  echo #ifndef THREADS_PER_BLOCK
  echo #define THREADS_PER_BLOCK 672
  echo #endif
  echo // SUBBLOCK_SIZE gives size of the sub-block as given in our article. It is calculated as ratio of DATA_SIZE and THREADS_PER_BLOCK rounded up so SUBBLOCK_SIZE=DATA_SIZE/THREADS_PER_BLOCK=2.6666, rounding up gives SUBBLOCK_SIZE=3;
  echo #ifndef SUBBLOCK_SIZE
  echo #define SUBBLOCK_SIZE 3
  echo #endif
  echo // Number of device 
  echo #ifndef DEVICE
  echo #define DEVICE 0
  echo #endif
  echo #ifndef OUTPUTWRITE 
  echo #define OUTPUTWRITE true
  echo #endif
  echo // Number of columns in shared memory.
  echo #ifndef SM_Columns 
) > C:\PFB\low-level\PFB_radioastronomie\PFB_radioastronomie\params.h
:: ) is not supported therefore last two lines have to be added this way.
 echo #define SM_Columns (DATA_SIZE / WARP - TAPS + 1) >> C:\PFB\low-level\PFB_radioastronomie\PFB_radioastronomie\params.h
 echo #endif >> C:\PFB\low-level\PFB_radioastronomie\PFB_radioastronomie\params.h
:: 0.2 Build vsproject $taps $channels $spectra
"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin\MSBuild.exe" C:\PFB\low-level\PFB_radioastronomie\PFB_radioastronomie.sln /property:Configuration=Debug 
:: 1. Run exe File
cd C:\PFB\low-level\PFB_radioastronomie\x64\Debug
PFB_radioastronomie.exe 
:: 2. Read timestamp from timestamp.txt
cd C:\Users\b98\PFB_OUTPUT
:: @ping -n 10 localhost> nul
if exist timestamp.txt set /p timestamp=<timestamp.txt
cd %timestamp%
:: 2. Read settings from setting file and start python script from Ubuntu App with $timestamp $taps $channels $spectra
for /f "tokens=1,2,3,4 delims==;" %%G in (settings) do (
	set %%G=%%H
	wsl python /root/pfb.py %timestamp% %%I %%H %%J
)
:: Compare output
wsl python /root/filecomparison.py %timestamp%
cd C:\