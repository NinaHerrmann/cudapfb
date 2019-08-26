
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <cufft.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <stddef.h>
#include <time.h>
#include "timer.cuh"
#include "Header.h"
#include <string>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "device_launch_parameters.h"
#include <stdlib.h>

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

template<typename T>
__device__ __forceinline__ T ldg(const T* ptr) {
#if __CUDA_ARCH__ >= 350
	return __ldg(ptr);
#else
	return *ptr;
#endif
}

__global__ void CPF_Fir_shared_32bit(float* d_data, float* d_spectra, float* d_coeff, int nChannels) {

	float ftemp;
	int  localId, warpId;
	int tx = threadIdx.x;
	__shared__ float s_data[1024];
	__shared__ float s_coeff[TAPS * 32];


	// The Id of the warp
	warpId = ((int)threadIdx.x / 32);
	// Id of thread within the warp
	localId = threadIdx.x - warpId * 32;
	// Number of rows which can be processed by one block 1024 threads so data block 32 * 32 coeff block 32 * taps
	int rows = 32 - TAPS;

	// LOad shared data from the point in global memory where localId + nChannels * waarp id + der y offset der rows plus der x offset der channels.

	s_data[tx] = d_data[localId + nChannels * warpId + rows * blockIdx.y * nChannels + 32 * blockIdx.x];
	// Load shared coeff from the point in global memory where localId + nChannels * waarp id + offset der channels
	if (tx < TAPS * 32) {
		s_coeff[tx] = ldg(&d_coeff[warpId * nChannels + localId + blockIdx.x * 32]);
	}
	__syncthreads();
	// In Case the index of global memory where we write to is smaller than the datasize do nothing.
	// in case the thread tx ist kleiner als die anzahl der elemente die wir pro block verarbeiten können.
	if (((localId + 32 * warpId + rows * blockIdx.y * nChannels + 32 * blockIdx.x) < (nChannels * NSPECTRA)) && (tx) < (rows * 32)) {
		//if (localId + 32 * warpId + rows * blockIdx.y * nChannels + 32 * blockIdx.x == 0) printf("check %d, %d, %d, %d \n", blockIdx.y, blockIdx.x, tx, localId + nChannels * warpId + rows * blockIdx.y * nChannels + 32 * blockIdx.x);

		ftemp = 0.0f;
		for (int j = 0; j < TAPS; j++) {
			// Wir haben nur noch den Block im shared memory also lade immer mit +32 als multiplikator pro iteration
			ftemp += s_coeff[j * 32 + localId] * (s_data[tx + j * 32]);
		} 
		// Schreibe das ergebniss zurück an den global index
		d_spectra[localId+ nChannels * warpId + rows * blockIdx.y * nChannels + 32 * blockIdx.x] = ftemp;
	}

	
}

__global__ void CPF_Fir_shared_32bit_Offset(float* d_data, float* d_spectra, float* d_coeff, int nChannels, int Offset) {

	float ftemp;
	int  localId, warpId;
	int tx = threadIdx.x;
	__shared__ float s_data[1024];
	__shared__ float s_coeff[TAPS * 32];


	// The Id of the warp
	warpId = ((int)threadIdx.x / 32);
	// Id of thread within the warp
	localId = threadIdx.x - warpId * 32;
	// Number of rows which can be processed by one block 1024 threads so data block 32 * 32 coeff block 32 * taps
	int rows = 32 - TAPS;

	// LOad shared data from the point in global memory where localId + nChannels * waarp id + der y offset der rows plus der x offset der channels.
	if (localId + nChannels * warpId + rows * blockIdx.y * nChannels + 32 * blockIdx.x + Offset < (NSPECTRA +TAPS-1)*nChannels) {
		s_data[tx] = d_data[localId + nChannels * warpId + rows * blockIdx.y * nChannels + 32 * blockIdx.x + Offset];
	}
	// Load shared coeff from the point in global memory where localId + nChannels * waarp id + offset der channels
	if (tx < TAPS * 32) {
		s_coeff[tx] = ldg(&d_coeff[warpId * nChannels + localId + blockIdx.x * 32]);
	}
	__syncthreads();
	// In Case the index of global memory where we write to is smaller than the datasize do nothing.
	// in case the thread tx ist kleiner als die anzahl der elemente die wir pro block verarbeiten können.
	if (((localId + 32 * warpId + rows * blockIdx.y * nChannels + 32 * blockIdx.x+ Offset) < (nChannels * NSPECTRA)) && (tx) < (rows * 32)) {
		//if (localId + 32 * warpId + rows * blockIdx.y * nChannels + 32 * blockIdx.x == 0) printf("check %d, %d, %d, %d \n", blockIdx.y, blockIdx.x, tx, localId + nChannels * warpId + rows * blockIdx.y * nChannels + 32 * blockIdx.x);

		ftemp = 0.0f;
		for (int j = 0; j < TAPS; j++) {
			// Wir haben nur noch den Block im shared memory also lade immer mit +32 als multiplikator pro iteration
			ftemp += s_coeff[j * 32 + localId] * (s_data[tx + j * 32]);
		}
		// Schreibe das ergebniss zurück an den global index
		d_spectra[localId + nChannels * warpId + rows * blockIdx.y * nChannels + 32 * blockIdx.x+ Offset] = ftemp;
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
int main(int argc, char* argv[]) {
	std::size_t ntaps, nchans, nspectra;
	// In case no arguments are given, it is assumed that the parameter from the params.h should be used.
	ntaps = TAPS;
	nchans = CHANNELS;
	nspectra = NSPECTRA;
	int input_size = (nspectra + ntaps - 1) * nchans;
	int coeff_size = ntaps * nchans;
	//---------> Host Memory allocation

	float* h_a;
	float* h_coeff;

	// call malloc to allocate that appropriate number of bytes for the array

	h_a = (float*)malloc(sizeof(float) * ((NSPECTRA + TAPS - 1) * CHANNELS));
	h_coeff = (float*)malloc(sizeof(float) * (TAPS * CHANNELS));
	/*float h_a[(NSPECTRA + TAPS - 1) * CHANNELS] = { 0 };
	float h_coeff[TAPS * CHANNELS] = { 0 };*/
	float* dev_a = 0;
	float* dev_coeff = 0;
	float* dev_a2 = 0;
	(cudaMalloc((void**)& dev_a, input_size * sizeof(float)));
	(cudaMalloc((void**)& dev_a2, input_size * sizeof(float)));
	(cudaMalloc((void**)& dev_coeff, coeff_size * sizeof(float)));
	srand(1);
	for (int n = 0; n < input_size; n++) {
		h_a[n] = 1;
	}
	for (int n = 0; n < coeff_size; n++) {
		h_coeff[n] = 1;
	}
	double FIR = 0.0;
	GpuTimer timer; // if set before set device getting errors - invalid handle  
	(cudaMemcpy(dev_coeff, h_coeff, coeff_size * sizeof(float), cudaMemcpyHostToDevice));
	(cudaMemcpy(dev_a, h_a, input_size * sizeof(float), cudaMemcpyHostToDevice));
	for (int n = 0; n < input_size; n++) {
		h_a[n] = 0;
	}

	(cudaMemcpy(dev_a2, h_a, input_size * sizeof(float), cudaMemcpyHostToDevice));
	int nCUDAblocks_x = (int)ceil((float)CHANNELS / 32);
	int nCUDAblocks_y = (int)ceil((nspectra + TAPS - 1) / (32 - TAPS));
	//printf("here");
	for (int i = 0; i < 1; i++) {

		printf("(%d;%d;%d), (%d;%d;%d), \n", nCUDAblocks_x, nCUDAblocks_y, 1, THREADS_PER_BLOCK, 1, 1);

		if (TIMER) timer.Start();
		int leftblocks = 0;
		if (nCUDAblocks_y > 65535) {
			leftblocks = nCUDAblocks_y - 65535;
			nCUDAblocks_y = 65535;
		}
		dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, 1); //nCUDAblocks_y goes through spectra
		dim3 blockSize(1024, 1, 1);                //nCUDAblocks_x goes through channels
		CPF_Fir_shared_32bit << <gridSize, blockSize >> > (dev_a, dev_a2, dev_coeff, nchans);

		if (leftblocks > 0) {
			printf("(%d;%d;%d), (%d;%d;%d), \n", nCUDAblocks_x, nCUDAblocks_y, 1, THREADS_PER_BLOCK, 1, 1);

			dim3 gridSize2(nCUDAblocks_x, 65535, 1);
			CPF_Fir_shared_32bit_Offset << <gridSize2, blockSize>> > (dev_a, dev_a2, dev_coeff, nchans, 67107840);
			leftblocks = leftblocks - 65535;
		}
		if (leftblocks > 0) {
			printf("(%d;%d;%d), (%d;%d;%d), \n", nCUDAblocks_x, nCUDAblocks_y, 1, THREADS_PER_BLOCK, 1, 1);

			dim3 gridSize3(nCUDAblocks_x, 65535, 1);
			//CPF_Fir_shared_32bit_Offset << <gridSize3, blockSize >> > (dev_a, dev_a2, dev_coeff, nchans, 67107840 *2);
			leftblocks = leftblocks - 65535;

		}
		if (leftblocks > 0) {
			printf("(%d;%d;%d), (%d;%d;%d), \n", nCUDAblocks_x, nCUDAblocks_y, 1, THREADS_PER_BLOCK, 1, 1);

			dim3 gridSize4(nCUDAblocks_x, 65535, 1);
			//CPF_Fir_shared_32bit_Offset << <gridSize4, blockSize >> > (dev_a, dev_a2, dev_coeff, nchans, 67107840 *3);
			leftblocks = leftblocks - 65535;

		}
		if (leftblocks > 0) {
			printf("(%d;%d;%d), (%d;%d;%d), \n", nCUDAblocks_x, nCUDAblocks_y, 1, THREADS_PER_BLOCK, 1, 1);

			dim3 gridSize5(nCUDAblocks_x, 65535, 1);
			//CPF_Fir_shared_32bit_Offset << <gridSize5, blockSize >> > (dev_a, dev_a2, dev_coeff, nchans, 67107840 *4);
			leftblocks = leftblocks - 65535;
		}

		if (TIMER) {
			timer.Stop();
			FIR += timer.Elapsed();
		}
		printf("%f\n", FIR);
		FIR = 0.0;
		cudaDeviceSynchronize();
	}
	(cudaMemcpy(h_a, dev_a2, input_size * sizeof(float), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	for (int i = 0; i < NSPECTRA*nchans; i++) {
		//if (h_a[i] != 0.00) {}
		//else { printf("%f, %d\n", h_a[i], i); break; }
		//printf("%f,", h_a[i]);
	}
	return 0;
}