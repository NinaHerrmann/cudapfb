	#include <cuda.h>
	#include <omp.h>
	#include <stdlib.h>
	#include <math.h>
	#include <array>
	#include <vector>
	#include <sstream>
	#include <chrono>
	#include <curand_kernel.h>
	#include <limits>
	#include <memory>
	#include <cstddef>
	#include <type_traits>
	
	
	#include "../include/musket.cuh"
	#include "../include/pfb64_0.cuh"

	
	struct Init_map_in_place_array_functor{
		
		Init_map_in_place_array_functor(){}
		
		~Init_map_in_place_array_functor() {}
		
		__device__
		auto operator()(float x){
			curandState_t curand_state; // performance could be improved by creating states before
			size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			curand_init(clock64(), id, 0, &curand_state);
			return static_cast<float>((curand_uniform(&curand_state) * (1.0f - 0.0f) + 0.0f));
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	struct FIR_map_in_place_array_functor{
		
		FIR_map_in_place_array_functor(){}
		
		~FIR_map_in_place_array_functor() {}
		
		__device__
		auto operator()(float const* __restrict__ d_data, float* d_spectra, float const* __restrict__ d_coeff){
			int Idx = ((blockIdx.x) * blockDim.x) + threadIdx.x + (64 * blockIdx.x);
			int tx = threadIdx.x;

			__shared__ float s_data[2048];
			__shared__ float s_coeff[1024];

			s_data[tx] = ldg(&d_data[Idx]);
			s_data[tx+1024] = ldg(&d_data[Idx+1024]);
	

			if (tx < 64 * 16) {
				s_coeff[tx] = ldg(&d_coeff[tx]);
			}

			__syncthreads();
			float ftemp = 0.0f;
			if (threadIdx.x < (2048 - (15 * 64))) {
				if (Idx < 64 * 2097152) {
					for (int i = 0; i < 16; i++) {
						if (i==0 && blockIdx.x == 1 && threadIdx.x < 2) printf(" coeff %f, data %d  \n", s_coeff[tx + (i * 64)], s_data[tx + (i * 64)]);

						ftemp += s_coeff[tx + (i * 64)] * (s_data[tx + (i * 64)]);
					}

					d_spectra[Idx] = ftemp;
				}
			}
			ftemp = 0.0f;
			if (threadIdx.x+1024 < (2048 - (15 * 64))) {
				if (Idx+1024 < 64 * 2097152) {
					for (int i = 0; i < 16; i++) {
						ftemp += s_coeff[(tx + (i * 64))+1024] * (s_data[(tx + (i * 64))+1024]);
					}
					d_spectra[Idx+1024] = ftemp;
				}
			}
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	struct Combine_map_index_in_place_array_functor{
		
		Combine_map_index_in_place_array_functor(){}
		
		~Combine_map_index_in_place_array_functor() {}
		
		__device__
		auto operator()(int Ai, float T){
			return (0.0f + );
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		int j;
		int i;
		
	};
	
	
	
	
	
	
	
int main(int argc, char** argv) {
	mkt::init();
		
		
	mkt::sync_streams();
	std::chrono::high_resolution_clock::time_point complete_timer_start = std::chrono::high_resolution_clock::now();
		
	mkt::DArray<float> input(0, 268435952, 268435952, 0.0f, 1, 0, 0, mkt::DIST, mkt::COPY);
	mkt::DArray<float> output(0, 268435952, 268435952, 0.0f, 1, 0, 0, mkt::DIST, mkt::COPY);
	mkt::DArray<float> coeff(0, 1024, 1024, 0.0f, 1, 0, 0, mkt::DIST, mkt::COPY);
		
	Init_map_in_place_array_functor init_map_in_place_array_functor{};
	FIR_map_in_place_array_functor fIR_map_in_place_array_functor{};
	Combine_map_index_in_place_array_functor combine_map_index_in_place_array_functor{};
		
	int ntaps = 16;
	int nchans = 64;
	int nspectra = 2097152;
	mkt::map_in_place<float, Init_map_in_place_array_functor>(input, init_map_in_place_array_functor);
	mkt::map_in_place<float, Init_map_in_place_array_functor>(coeff, init_map_in_place_array_functor);
	output = (input);
	mkt::sync_streams();
	std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
// TODO Pass arguments! + threads are listed in functor? check that.
//const int gpu_elements = a.get_size_gpu();
		//		int threads = gpu_elements < 1024 ? gpu_elements : 1024; // nextPow2
		//		int blocks = (gpu_elements + threads - 1) / threads;
	mkt::map_in_place<float, FIR_map_in_place_array_functor>(input, fIR_map_in_place_array_functor);
	int log2p = 6;
	int log2size = 28;
	float j = 0;
	for(int i = 0; ((i) < (log2p)); i++){
		combine_map_index_in_place_array_functor.j = (j);
		combine_map_index_in_place_array_functor.i = (output);
		mkt::map_index_in_place<float, Combine_map_index_in_place_array_functor>(input, combine_map_index_in_place_array_functor);
	}
	mkt::sync_streams();
	std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
	double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
		
	mkt::sync_streams();
	std::chrono::high_resolution_clock::time_point complete_timer_end = std::chrono::high_resolution_clock::now();
	double complete_seconds = std::chrono::duration<double>(complete_timer_end - complete_timer_start).count();
	printf("Complete execution time: %.5fs\n", complete_seconds);
		
	printf("Execution time: %.5fs\n", seconds);
	printf("Threads: %i\n", 0);
	printf("Processes: %i\n", 1);
		
	return EXIT_SUCCESS;
}
