#include "device_launch_parameters.h"
#include <stdlib.h>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stddef.h>
#include <cstdint>
#include <time.h>
#include "params.h"
#include "debug.h"
#include <string>
#include "CriticalPolyphaseFilterbank.h"
#include "filehelper.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_profiler_api.h>
#include <cudaProfiler.h>

__global__ void print_thx() {
	printf("Thread: %d\n", threadIdx.x);
}

int CPF_Max_columns_in_memory_SM(int nTaps, int nChannels) {
	long int maxColumns, maxgrid_y, itemp;

	size_t free_mem, total_mem;
	cudaDeviceProp devProp;

	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaGetDeviceProperties(&devProp, 0));
	maxgrid_y = devProp.maxGridSize[1];
	cudaMemGetInfo(&free_mem, &total_mem);
	if (DEBUG) printf("\nMemory free : %zd, total : %zd \n", free_mem, total_mem);
	// ? Calculates the maximum number of column/row by subtracting the memory needed for the coeff and the additional taps and dividing by the space needed for a column/row.
	// TODO WTF * 3 for divison
	maxColumns = ((long int)free_mem - nChannels * sizeof(float)*nTaps - (nTaps - 1)*nChannels * sizeof(float)) / (3.0 * sizeof(float)*nChannels);
	// TODO From the maximum columns we can only use 90%?
	maxColumns = (int)(maxColumns*0.9);
	// TODO : in case the maxgrid size multiplied by the shared memory we want per block is smaller than the maximum number of columns? --> wie hngt das zusammen 
	if (maxgrid_y*SM_Columns < maxColumns) maxColumns = maxgrid_y * SM_Columns;
	itemp = (int)(maxColumns / SM_Columns);
	maxColumns = itemp * SM_Columns;
	return(maxColumns);
}

bool checkinput(int argc, char *argv[]) {
	printf("Number of Arguments : %d \n", argc);
	if (argc != 4) {
		printf("You supplied an invalid number of arguments. Must be number of taps, number of channels, and number of spectra.\n");
		exit(0);
	}
	bool checker = true;
	errno = 0;
	char *endptr1, *endptr2, *endptr3;
	long int ntaps = strtol(argv[1], &endptr1, 10);
	long int nchans = strtol(argv[2], &endptr2, 10);
	long int nspectra = strtol(argv[3], &endptr3, 10);
	// Check whether arguments supplied for number of taps is a number.
	if (endptr1 == argv[1]) {
		printf("Invalid number for number of taps: %%s \n", argv[1]);
		checker = false;
	}
	else if (*endptr1) {
		printf("Trailing characters after number of taps:  %%s \n", argv[1]);
		checker = false;
	}
	else if (errno == ERANGE) {
		printf("Number of taps of range:  %%s \n", argv[1]);
		checker = false;
	}
	// Check whether arguments supplied for number of channels is a number.
	if (endptr2 == argv[2]) {
		printf("Invalid number for number of channels: %%s \n", argv[2]);
		checker = false;
	}
	else if (*endptr2) {
		printf("Trailing characters after number of channels:  %%s \n", argv[2]);
		checker = false;
	}
	else if (errno == ERANGE) {
		printf("Number of channels of range:  %%s \n", argv[2]);
		checker = false;
	}
	// Check whether arguments supplied for number of spectra is a number.
	if (endptr3 == argv[3]) {
		printf("Invalid number for number of spectra: %%s \n", argv[3]);
		checker = false;
	}
	else if (*endptr3) {
		printf("Trailing characters after number of spectra:  %%s \n", argv[3]);
		checker = false;
	}
	else if (errno == ERANGE) {
		printf("Number of spectra of range:  %%s \n", argv[3]);
		checker = false;
	}
	// Stop execution in case one of the params is not of right type.
	if (!checker) { exit(0); }
	// TODO any semantic checks required?
	// int ntaps = atoi(argv[1]);
	// int nchans = atoi(argv[2]);
	// int nspectra = atoi(argv[3]);
	return checker;
}

/*
* Implementing a short security check reassures whether big input data should really be processed with the CMDWrite/Write option because this will take a long time. 
*/
void checkwritecorrectness(int input_size) {
	if (input_size > 10000 & WRITE || input_size > 10000 & CMDWRITE) {
		char a_word = 'n';
		char yes = 'y';
		printf("Your input size is %d floats, WRITE is set to %s and CMDWRITE is set to %s. \nThe execution might take a while.\nIf you want to continue press y.\n", input_size, WRITE ? "true" : "false", CMDWRITE ? "true" : "false");
		scanf("%c", &a_word);
		if (a_word != yes) {
			exit(1);
		}
	}
}
/*
* This is a very basic memory check. Actually the calculation is relatively basic. We habe for the FIR Calculation floating numbers:
* Input : (nspectra + ntaps - 1) * nchans;
* Coeff : ntaps * nchans;
* Output : (nspectra + ntaps - 1) * nchans;
* For the FFT we have the output of the FIR Calculation so we do not need an extra space but we do have the output Which requires GPU Memory we have cufft Complex as type:
* Output : (nspectra + ntaps - 1) * nchans;
// ! Remark - I think cufft requires output size like that but we have to check that! 
* This subsumes to 3 * ( (nspectra + ntaps - 1) * nchans) + ntaps * nchans; = nchans(3 * (nspectra + ntaps - 1) + ntaps) 
* (For simplicity it is split in the function.)
*/
void checkmemory(std::size_t nspectra, std::size_t ntaps, std::size_t nchans, int & nRepeats) {
	long gpu_space_req = ((nchans * (2*(nspectra + ntaps - 1) + ntaps)) * sizeof(float)) + ((nchans * (nspectra + ntaps - 1)) * sizeof(cufftComplex));
	size_t free_mem, total_mem;

	cudaMemGetInfo(&free_mem, &total_mem);
	if (DEBUG) printf("\nMemory free : %zd, total : %zd, space required : %ld\n", free_mem, total_mem, gpu_space_req);
	if (free_mem < gpu_space_req) {
		printf("We will need to split the data to progress it...\n");
	}
	else {
		printf("Everything suits in Memory :) \n\n");
	}
}


/*
* There exist two ways to set nTaps, nSpectra, and nChannels due to two different use cases.
* The primary use case is for the MPI to set nTaps, nSpectra, and nChannels flexible with every call to the polyphase Filterbank. For this purpose the
* parameter cannot be global and have to be passed to the CPF class.
*
* The secondary use case is discussing the performance of the program.
* In general a program using global variables is expected to be faster. For this purpose they are still part of the params.h.
* However, using the global variables in the program is WIP. To ease testing when no parameters are passed the parameter from the params.h is used.
* TODO: Evaluate whether a second project is required for global variables or whether a case distinction is sufficient.
*/
int main(int argc, char *argv[]) {
	std::size_t ntaps, nchans, nspectra;
	if (argc == 4) {
		// Checks if the given arguments are numerical. 
		checkinput(argc, argv);
		ntaps = atoi(argv[1]);
		nchans = atoi(argv[2]);
		nspectra = atoi(argv[3]);
	}
	// In case no arguments are given, it is assumed that the parameter from the params.h should be used.
	else {
		ntaps = TAPS;
		nchans = CHANNELS;
		nspectra = NSPECTRA;
	}

	// TODO: Check Memory before starting.
	if (DEBUG) {
		printf("\n\t\t-----Welcome to the PFB Implementation for GPGPU -----\n\n");
		printf("This program runs with %zd Channels, %zd Taps, and %zd Spectra.\n", nchans, ntaps, nspectra);
		printf("Debug variables are set to: Debug: %s, CMDWRITE: %s, and WRITE: %s\n", DEBUG ? "true" : "false", CMDWRITE ? "true" : "false", WRITE ? "true" : "false");
	}
	int input_size = (nspectra + ntaps - 1)*nchans;
	// TODO This is actually not true. We have only nspectra * (nchans/2 +1) Numbers and Ewan even suggested to cut one number since it is useless. which would result in (nspectra * nchans/2) 
	// TODO However, I guess the cufft library requires the space anyway so we better occupie it.
	int output_size = nspectra * nchans;
	int coeff_size = ntaps * nchans;
	int maxColumns, Sremainder, nRepeats, itemp, Spectra_to_run;

	if (!BENCHMARK) checkwritecorrectness((nspectra + ntaps - 1)*nchans);

	if (!BENCHMARK) checkmemory(nspectra, ntaps, nchans, nRepeats);


	maxColumns = CPF_Max_columns_in_memory_SM(ntaps, nchans); // Maximum number of columns which fit into memory
	// In case we have more spectra than maximum number of columns we have to repeat the process.
	nRepeats = (int)(nspectra / maxColumns);
	// In case the the maximum number of columns is not a multiple or dividor of number of spectra we have a remainder part.
	Sremainder = nspectra - nRepeats * maxColumns;

	// itemp = anzahl der blöcke
	itemp = (int)(Sremainder / SM_Columns);
	if ((Sremainder - itemp * SM_Columns) > 0) itemp++;
	Spectra_to_run = itemp * SM_Columns; // Since shared memory kernel has fixed number of columns it loads we need to process more spectra then needed.
		//---------> Channels
	int nCUDAblocks_x = (int)nchans / THXPERWARP; //Head size
	int Cremainder = nchans - nCUDAblocks_x * THXPERWARP; //Tail size
	if (DEBUG) printf("Cremainder %d, nCUDAblocks_x %d nChans %zd Warp %d \n", Cremainder, nCUDAblocks_x, nchans, THXPERWARP);
	if (Cremainder > 0) { printf("Number of channels must be divisible by 32"); exit(2); }

	//---------> Host Memory allocation
	std::vector<float> h_input;
	std::vector<float> h_output;
	std::vector<float> h_coeff;
	float2 * h_output2;
	h_output2 = (float2*)malloc(output_size * sizeof(float2));

	int c_output_size = nchans * (ntaps + nspectra);

	if (DEBUG) printf("\nHost memory memset...\t\t");
	std::fill(h_output.begin(), h_output.end(), (float) 0);
	if (DEBUG) printf("done.");

	if (DEBUG) printf("\nRandom data set...\t\t");
	srand(1);
	
	unsigned long timestamp = (unsigned long)time(NULL);
	char buffer[12];
	const char * timestampchar = ultoa(timestamp, buffer, 10);
	generateandwriteinput(ntaps, nchans, nspectra, timestamp, h_input, input_size, coeff_size, h_coeff);

	// CPF_dynamic_setting_calculation();
	int devCount = 0;
	checkCudaErrors(cudaSetDevice(0));
	cudaDeviceProp devProp;
	size_t free_mem, total_mem;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	if (CMDWRITE) {
		printf("\nThere are %d devices.", devCount);
		for (int i = 0; i < devCount; i++) {
			checkCudaErrors(cudaGetDeviceProperties(&devProp, i));
			printf("\n\t Using device:\t\t\t%s\n", devProp.name);
			printf("\n\t Max grid size:\t\t\t%d\n", devProp.maxGridSize[1]);
			printf("\n\t Shared mem per block:\t\t%zd\n", devProp.sharedMemPerBlock);
			printf("\n\t Total global Memory:\t\t%zd\n", devProp.totalGlobalMem);
			printf("\n\t Total constant Memory :\t%zd\n", devProp.totalConstMem);
			printf("\n\t Device Overlap :\t%d\n", devProp.deviceOverlap);
			printf("\n\t Maximum amount of Memory allocated by MallocPitch:\t%zd\n", devProp.memPitch);
			printf("\n\t Max grid size:\t\t\t%d, %d, %d\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
			printf("\n\t SM:\t\t\t%d\n", devProp.multiProcessorCount);
			printf("\n\t Minor:\t\t\t%d\n", devProp.minor);
			printf("\n\t Major:\t\t\t%d\n", devProp.major);
		}
	}

	//---------> Measurements
	double transfer_in = 0.0, transfer_out = 0.0, fir_time = 0.0, fft_time = 0.0, copy_input = 0.0, device_allocation = 0.0;
	GpuTimer timer; // if set before set device getting errors - invalid handle  

	//---------> GPU Memory allocation
	std::size_t d_input_size = nchans * (maxColumns + ntaps - 1);
	std::size_t d_output_size = nchans * maxColumns;

	if (DEBUG) printf("Device memory allocation... Sremainder %d \t\t", Sremainder);
	timer.Start();
	// ! Change size n nRepeats -- How to do partial Memcpy.
	thrust::device_vector<float> d_input((nspectra + ntaps - 1)*nchans, (float)0.0);
	thrust::device_vector<float> d_output((nspectra + ntaps - 1)*nchans, (float)0.0);
	thrust::device_vector<float> d_coeff(coeff_size, (float)0.0);
	cufftComplex * d_output2;
	checkCudaErrors(cudaMalloc((void **)&d_output2, sizeof(cufftComplex)*c_output_size));
	timer.Stop();
	device_allocation += timer.Elapsed();
	if (DEBUG) printf("done in %g ms.\n", timer.Elapsed());

	//---------> Transfer coefficients to the device

	if (DEBUG) printf("\nCopy coefficients HTD   ...\t\t");
	timer.Start();
	d_coeff = h_coeff;
	timer.Stop();
	transfer_in += timer.Elapsed();
	if (DEBUG) printf("done in %g ms.\n\n", timer.Elapsed());
	CriticalPolyphaseFilterbank CriticalPolyphaseFilterbank(nchans, ntaps, d_coeff);
	timer.Start();
	d_input = h_input;
	timer.Stop();
	transfer_in += timer.Elapsed();
	//---------> Polyphase filter
	// TODO : Todecide : I think we do not need this in the MPIfR - we never have datachunks which are so big that they require multiple calls.
	for (int r = 0; r < nRepeats; r++) {
		//-----> Compute Polyphase on the chunk
		//---------> Set the prefered Memory.
		cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		if (DEBUG) printf("Starting the benchmark...\n");
		// TODO ----------------------------------------------------------------------------------
		CriticalPolyphaseFilterbank.process(d_input, d_output, timestampchar, Spectra_to_run, fir_time, fft_time, d_output2, 0);
		//-----> Copy chunk of output data to host
		timer.Start();
		checkCudaErrors(cudaMemcpy(&h_output[r*output_size], d_output2, d_output_size * sizeof(float), cudaMemcpyDeviceToHost));
		timer.Stop();
		transfer_out += timer.Elapsed();
	}
	if (Sremainder > 0) {
		//-----> Compute Polyphase on the chunk
		cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		if (DEBUG) printf("Starting the benchmark...\n");
		if (DEBUG) cudaProfilerStart();
		CriticalPolyphaseFilterbank.process(d_input, d_output, timestampchar, Spectra_to_run, fir_time, fft_time, d_output2, 0);
		if (DEBUG) cudaProfilerStop();

		//-----> Copy chunk of output data to host
		timer.Start();
		// TODO:: Actually, we need only half of all entries (due to Ewan) 
		checkCudaErrors(cudaMemcpy(&h_output2[nRepeats * d_output_size], d_output2, Sremainder*nchans * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
		timer.Stop();
		transfer_out += timer.Elapsed();
	}
	
	if (WRITE) writeoutput(timestamp, h_output2, timestampchar, nspectra, fir_time, fft_time, transfer_in, transfer_out, nchans, ntaps, 0, THREADS_PER_BLOCK, DATA_SIZE);
	
	//---------> error check -----
	checkCudaErrors(cudaGetLastError());

	//---------> Freeing allocated resources
	h_input.clear();
	h_output.clear();
	h_coeff.clear();

	h_input.shrink_to_fit();
	h_output.shrink_to_fit();
	h_coeff.shrink_to_fit();

	d_output.clear();
	d_coeff.clear();
	d_input.clear();

	d_output.shrink_to_fit();
	d_coeff.shrink_to_fit();
	d_input.shrink_to_fit();
	checkCudaErrors(cudaFree(d_output2));

	printf("\n\nCalculations are finished:\nData transfer time %0.3f ms\nTransfer in %0.3f GPU allocation %f\n", transfer_in + transfer_out, transfer_in, device_allocation);

	/* cudaError_t cudaStatus = cudaSuccess;
	// cudaError_t cudaStatus = CriticalPolyphaseFilterbank(ntaps, nchans, filter_coefficients, input);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CriticalPolyphaseFilterbank failed!");
        return 1;
    }
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
	// TODO: check whether this has to be removed for streaming.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }*/
	
    return 0;
}

