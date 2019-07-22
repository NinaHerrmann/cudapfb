
#include "cuda_runtime.h"
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
std::vector<float> h_input(16384); //Size:16384, localSize: 16384, gpuSize: 16384
std::vector<float> h_input_glbcopy(16384);
float* d_input0_loccopy;
float* d_input0_glbcopy;
//Declare everything for overflow
std::vector<float> h_overflow(512); //Size:512, localSize: 512, gpuSize: 512
std::vector<float> h_overflow_glbcopy(512);
float* d_overflow0_loccopy;
float* d_overflow0_glbcopy;
//Declare everything for coeff
std::vector<float> h_coeff(512); //Size:480, localSize: 480, gpuSize: 480
std::vector<float> h_coeff_glbcopy(512);
float* d_coeff0_loccopy;
float* d_coeff0_glbcopy;

//------ Kernel Functions -----
__global__ void init_MapIndexInPlaceSkeleton_array(int rowOffset, float* y) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	float x = ((index + rowOffset) % 100) * 0.01;
	y[index] = (x);
}
// h_input_xOffset + 0, 0, (ntaps), (nchans), (nspectra), d_input0, d_output0
__global__ void FIR_MapIndexInPlaceSkeleton_array(int a, int y, int taps, int rowOffset, int nspectra, float* p_input, float* p_coeff, float* d_output) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int channels = rowOffset;
	//printf("%d,", index);
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



//------ Main Function -----
int main(int argc, char** argv) {
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
	int h_overflow_xOffset = 0;
	int h_coeff_xOffset = 0;

	// Allocate DataStructures and
	//no values present on host to initialize devices (data will be initialised directly on devices)
	//Allocate Structures
	tmp_size_t = 16384 * sizeof(float);
	float* d_input0;
	cudaSetDevice(0);
	cudaMalloc(&d_input0, tmp_size_t);
	//no values present on host to initialize devices (data will be initialised directly on devices)
	//Allocate Structures
	tmp_size_t = 512 * sizeof(float);
	float* d_overflow0;
	cudaSetDevice(0);
	cudaMalloc(&d_overflow0, tmp_size_t);
	//no values present on host to initialize devices (data will be initialised directly on devices)
	//Allocate Structures
	tmp_size_t = 512 * sizeof(float);
	float* d_coeff0;
	cudaSetDevice(0);
	cudaMalloc(&d_coeff0, tmp_size_t);

	int ntaps = 16;
	int nchans = 32;
	int nspectra = 497;
	if (16384 <= maxThreadsPerBlock) {
		numberOfBlocks = 1;
		numberOfThreads = 16384;
	}
	else {
		numberOfThreads = maxThreadsPerBlock;
		numberOfBlocks = ceil(16384.0 / numberOfThreads);
	}
	printf("Working on %d blocks and %d threads!\n", numberOfBlocks, numberOfThreads);
	//MapIndexInPlace Call
	cudaSetDevice(0);
	init_MapIndexInPlaceSkeleton_array << <numberOfBlocks, numberOfThreads, 0, streams[0] >> > (h_input_xOffset + 0, d_input0);

	fetch_size = 16384 * sizeof(float);

	cudaMemcpyAsync(&h_input[0], d_input0, fetch_size, cudaMemcpyDeviceToHost, streams[0]);
	h_input_glbcopy = h_input;
	std::ostringstream s4;
	s4 << "h_input: " << std::endl << "[";
	for (int i = 0; i < 511; i++) {
		s4 << h_input_glbcopy[i];
		s4 << "; ";
	}
	s4 << h_input_glbcopy[511] << "]" << std::endl;
	s4 << std::endl;
	// printf("%s", s4.str().c_str());

	if (512 <= maxThreadsPerBlock) {
		numberOfBlocks = 1;
		numberOfThreads = 512;
	}
	else {
		numberOfThreads = maxThreadsPerBlock;
		numberOfBlocks = ceil(512.0 / numberOfThreads);
	}
	printf("Working on %d blocks and %d threads!\n", numberOfBlocks, numberOfThreads);
	//MapIndexInPlace Call
	cudaSetDevice(0);
	init_MapIndexInPlaceSkeleton_array << <numberOfBlocks, numberOfThreads, 0, streams[0] >> > (h_overflow_xOffset + 0, d_overflow0);
	if (512 <= maxThreadsPerBlock) {
		numberOfBlocks = 1;
		numberOfThreads = 512.0;
	}
	else {
		numberOfThreads = maxThreadsPerBlock;
		numberOfBlocks = ceil(512.0 / numberOfThreads);
	}
	printf("Working on %d blocks and %d threads!\n", numberOfBlocks, numberOfThreads);
	//MapIndexInPlace Call
	cudaSetDevice(0);
	init_MapIndexInPlaceSkeleton_array << <numberOfBlocks, numberOfThreads, 0, streams[0] >> > (h_coeff_xOffset + 0, d_coeff0);
	//Print array h_coeff

	tmp_size_t = 16384 * sizeof(float);
	float* d_output0;
	cudaSetDevice(0);
	cudaMalloc(&d_output0, tmp_size_t);
	std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
	if (16384 <= maxThreadsPerBlock) {
		numberOfBlocks = 1;
		numberOfThreads = 16384;
	}
	else {
		numberOfThreads = maxThreadsPerBlock;
		numberOfBlocks = ceil(16384.0 / numberOfThreads);
	}
	printf("Working on %d blocks and %d threads!\n", numberOfBlocks, numberOfThreads);
	//Create needed global copys
	//Make complete, global copy of input:
	// ? fetch_size = 16384*sizeof(int);
	// ? cudaSetDevice(0);
	// ? cudaMalloc(&d_input0, fetch_size);
	// ?  cudaMemcpyAsync(d_input0, &h_input[0], fetch_size, cudaMemcpyHostToDevice, streams[0]);

	//Make complete, global copy of coeff:
	// ? fetch_size = 480*sizeof(int);
	// ? cudaSetDevice(0);
	// ? cudaMalloc(&d_coeff0, fetch_size);
	// ? cudaMemcpyAsync(d_coeff0, &h_coeff[0], fetch_size, cudaMemcpyHostToDevice, streams[0]);

	//MapIndexInPlace Call
	cudaSetDevice(0);
	//																							int a, int y, int taps, int rowOffset, int* nspectra, int* p_input, int* p_coeff
	FIR_MapIndexInPlaceSkeleton_array << <numberOfBlocks, numberOfThreads, 0, streams[0] >> > (h_input_xOffset + 0, 0, (ntaps), (nchans), (nspectra), d_input0, d_coeff0, d_output0);
	std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
	double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
	printf("Took %f seconds \n", seconds);
	fetch_size = 16384 * sizeof(float);
	cudaMemcpy(&h_input[0], d_output0, fetch_size, cudaMemcpyDeviceToHost);
	//Show h_input as h_input_glbcopy
	//hostdata is up-to-date, just make copy_variablename accessible
	h_input_glbcopy = h_input;
	//Print array h_input
	std::ostringstream s0;
	s0 << "h_output: " << std::endl << "[";
	for (int i = 0; i < 16383; i++) {
		s0 << h_input_glbcopy[i];
		s0 << "; ";
	}
	s0 << h_input_glbcopy[16383] << "]" << std::endl;
	s0 << std::endl;
	printf("%s", s0.str().c_str());

	//Print array h_coeff
	fetch_size = 512 * sizeof(float);

	cudaMemcpyAsync(&h_coeff[0], d_coeff0, fetch_size, cudaMemcpyDeviceToHost, streams[0]);
	h_coeff_glbcopy = h_coeff;
	std::ostringstream s1;
	s1 << "h_coeff: " << std::endl << "[";
	for (int i = 0; i < 479; i++) {
		s1 << h_coeff_glbcopy[i];
		s1 << "; ";
	}
	s1 << h_coeff_glbcopy[479] << "]" << std::endl;
	s1 << std::endl;
	printf("%s", s1.str().c_str());

	// Free DataStructures
	cudaFree(d_input0);
	cudaFree(d_input0_loccopy);
	cudaFree(d_input0_glbcopy);
	cudaFree(d_overflow0);
	cudaFree(d_overflow0_loccopy);
	cudaFree(d_overflow0_glbcopy);
	cudaFree(d_coeff0);
	cudaFree(d_coeff0_loccopy);
	cudaFree(d_coeff0_glbcopy);

	// MPI Finalisation
	return EXIT_SUCCESS;
}

