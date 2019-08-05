
#include "cuda_runtime.h"
#include "timer.cuh"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <array>
#include <iostream>
#include <time.h>
#include <chrono>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <cuda.h>
#include <stdexcept>
#include <sstream>

// Global Constants
const size_t number_of_processes = 1;
const size_t number_of_gpus = 1;
// Global Variables
// Temporary Variables
size_t tmp_size_t = 0;
size_t fetch_size = 0;

//------ DataStructure Declarations -----
	//Declare everything for input
std::vector<float> h_input(250000960); //Size:250000960, localSize: 250000960, gpuSize: 250000960
std::vector<float> h_input_glbcopy(250000960);
float* d_input0_loccopy;
float* d_input0_glbcopy;

//Declare everything for coeff
std::vector<float> h_coeff(1024); //Size:480, localSize: 480, gpuSize: 480
std::vector<float> h_coeff_glbcopy(1024);
float* d_coeff0_loccopy;
float* d_coeff0_glbcopy;

//------ Kernel Functions -----
__global__ void print(float* y) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	printf("y at %d:%f\n", index, y[index]);
}
// h_input_xOffset + 0, 0, (ntaps), (nchans), (nspectra), d_input0, d_output0
__global__ void FIR_MapIndexInPlaceSkeleton_array(int a, int y, int taps, int rowOffset, int nspectra, float* p_input, float* p_coeff, float* d_output) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int channels = rowOffset;
	float newa = 0;
	for (int j = 0; ((j) < (taps)); j++) {
		if (((index + ((j) * (channels)))) <= (nspectra * channels)) {
			newa += (// TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed
				p_input[(index + ((j) * (channels)))]
				* // TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed
				p_coeff[(((j) * (channels)))]
				);
		}
		else {
			// Do nothing since we do not have further data
		}
	}
	d_output[index] = (newa);
}

void writefloattofilevector(std::size_t datasize, std::vector<float> data) {
	for (int n = 0; n < (datasize); n++) {
		float x = (rand() / (float)RAND_MAX);
		// printf("\n Write at place %d value %f", n , x);
		data[n] = x;
	}
}

//------ Main Function -----
int main(int argc, char** argv) {
	GpuTimer timer;
	srand(1);
	double init_datastructure = 0.0, fir = 0.0, fill_ds = 0.0, out_ds = 0.0;
	timer.Start();
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	int numberOfThreads, numberOfBlocks, maxThreadsPerBlock;
	int totalGlobalMemory = 0;
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		maxThreadsPerBlock = prop.maxThreadsPerBlock;
	}
	// Stream Initialisation	
	cudaStream_t streams[number_of_gpus];
	for (int k = 0; k < number_of_gpus; k++) {
		cudaSetDevice(k);
		cudaStreamCreate(&streams[k]);
	}

	int h_input_xOffset = 0;
	int h_coeff_xOffset = 0;
	timer.Stop();
	init_datastructure += timer.Elapsed();
	timer.Start();
	// Allocate DataStructures and
	// no values present on host to initialize devices (data will be initialised directly on devices)
	// Allocate Structures
	tmp_size_t = 250000960 * sizeof(float);
	float* d_input0;
	cudaSetDevice(0);
	cudaMalloc(&d_input0, tmp_size_t);
	float* d_output0;
	cudaSetDevice(0);
	cudaMalloc(&d_output0, tmp_size_t);
	for (int n = 0; n < (250000960); n++) {
		float x = (rand() / (float)RAND_MAX);
		// printf("\n Write at place %d value %f", n , x);
		h_input[n] = x;
	}

	cudaMemcpy(d_input0, &h_input[0], tmp_size_t, cudaMemcpyHostToDevice);
	// print << <1, 16 >> > (d_input0);
	//no values present on host to initialize devices (data will be initialised directly on devices)
	//Allocate Structures
	tmp_size_t = 1024 * sizeof(float);
	float* d_coeff0;
	cudaSetDevice(0);
	cudaMalloc(&d_coeff0, tmp_size_t);
	for (int n = 0; n < (1024); n++) {
		float x = (rand() / (float)RAND_MAX);
		// printf("\n Write at place %d value %f", n , x);
		h_coeff[n] = x;
	}
	cudaMemcpyAsync(d_coeff0, &h_coeff[0], tmp_size_t, cudaMemcpyHostToDevice, streams[0]);

	int ntaps = 16;
	int nchans = 64;
	int nspectra = 3906250;
	timer.Stop();
	fill_ds += timer.Elapsed();
	timer.Start();

	if (250000960 <= maxThreadsPerBlock) {
		numberOfBlocks = 1;
		numberOfThreads = 250000960;
	}
	else {
		numberOfThreads = maxThreadsPerBlock;
		numberOfBlocks = ceil(250000960.0 / numberOfThreads);
	}
	printf("Working on %d blocks and %d threads!\n", numberOfBlocks, numberOfThreads);

	//MapIndexInPlace Call
	cudaSetDevice(0);
	//	int a, int y, int taps, int rowOffset, int* nspectra, int* p_input, int* p_coeff
	FIR_MapIndexInPlaceSkeleton_array << <numberOfBlocks, numberOfThreads, 0, streams[0] >> > (h_input_xOffset + 0, 0, (ntaps), (nchans), (nspectra), d_input0, d_coeff0, d_output0);
	timer.Stop();
	fir += timer.Elapsed();
	timer.Start();
	fetch_size = 250000960 * sizeof(float);
	cudaMemcpy(&h_input[0], d_output0, fetch_size, cudaMemcpyDeviceToHost);
	//Show h_input as h_input_glbcopy
	//hostdata is up-to-date, just make copy_variablename accessible
	h_input_glbcopy = h_input;

	fetch_size = 1024 * sizeof(float);

	cudaMemcpyAsync(&h_coeff[0], d_coeff0, fetch_size, cudaMemcpyDeviceToHost, streams[0]);
	h_coeff_glbcopy = h_coeff;

	// Free DataStructures
	cudaFree(d_input0);
	cudaFree(d_input0_loccopy);
	cudaFree(d_input0_glbcopy);
	cudaFree(d_coeff0);
	cudaFree(d_coeff0_loccopy);
	cudaFree(d_coeff0_glbcopy);
	timer.Stop();
	out_ds += timer.Elapsed();
	printf("\n%f; %f; %f; %f", init_datastructure, fill_ds, fir, out_ds);

	// MPI Finalisation
	return EXIT_SUCCESS;
}

