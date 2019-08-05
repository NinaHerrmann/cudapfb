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
	
	
	
	const double PI = 3.141592653589793;
	
	//Complex::Complex() : x(), y() {}
	

	
	struct Init_map_in_place_array_functor{
		
		Init_map_in_place_array_functor(){}
		
		~Init_map_in_place_array_functor() {}
		
		__device__
		auto operator()(int x){
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
	struct Zero_fill_map_in_place_array_functor{
		
		Zero_fill_map_in_place_array_functor(){}
		
		~Zero_fill_map_in_place_array_functor() {}
		
		__device__
		auto operator()(int x){
			return static_cast<float>(0);
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	struct Bitrev_reorder_map_index_in_place_array_functor{
		
		Bitrev_reorder_map_index_in_place_array_functor(const mkt::DArray<float>& _input) : input(_input){}
		
		~Bitrev_reorder_map_index_in_place_array_functor() {}
		
		__device__
		auto operator()(int x, int i){
			int brev = 0;
			int __powf = 0;
			return // TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed
			input.get_data_local(((x) / std::pow2(2, (32 - (log2size)))))
			;
		}
	
		void init(int device){
			input.init(device);
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		int log2size;
		
		mkt::DeviceArray<float> input;
	};
	struct Fetch_map_index_in_place_array_functor{
		
		Fetch_map_index_in_place_array_functor(const mkt::DArray<float>& _input) : input(_input){}
		
		~Fetch_map_index_in_place_array_functor() {}
		
		__device__
		auto operator()(int i, float Ti){
			return // TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed
			input.get_data_local(std::pow((i), std::pow(2, (((log2size) - 1) - (counter)))))
			;
		}
	
		void init(int device){
			input.init(device);
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		int counter;
		int log2size;
		
		mkt::DeviceArray<float> input;
	};
	struct Combine_map_index_in_place_array_functor{
		
		Combine_map_index_in_place_array_functor(const mkt::DArray<float>& _input_double) : input_double(_input_double){}
		
		~Combine_map_index_in_place_array_functor() {}
		
		__device__
		auto operator()(int Index, float Ai){
			float newa = 0.0f;
			int b = ((Index) / std::pow(2, (((log2size) - 1) - (counter))));
			int b2 = 0;
			for(int l = 0; ((l) <= (counter)); l++){
				
				if(((b) == 1)){
				b2 = ((2 * (b2)) + 1);
				}
				 else {
						b2 = (2 * (b2));
					}
				b = ((b) / 2);
			}
			float temp = ((((2.0 * (pi)) / (Problemsize)) * (b2)) * std::pow(2, (((log2size) - 1) - (counter))));
			
			if(((Index) == std::pow(2, (((log2size) - 1) - (counter))))){
			newa = (// TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed
			input_double.get_data_local((Index))
			 + ((temp) * (Ai)));
			}
			 else {
					newa = ((Ai) + ((temp) * // TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed
					input_double.get_data_local((Index))
					));
				}
			return (newa);
		}
	
		void init(int device){
			input_double.init(device);
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		int counter;
		int log2size;
		double pi;
		int Problemsize;
		
		mkt::DeviceArray<float> input_double;
	};
	
	
	
	
	
	
	
	int main(int argc, char** argv) {
		mkt::init();
		
		
		mkt::sync_streams();
		std::chrono::high_resolution_clock::time_point complete_timer_start = std::chrono::high_resolution_clock::now();
		
		mkt::DArray<float> input(0, 16, 16, 0.0f, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<float> input_double(0, 16, 16, 0.0f, 1, 0, 0, mkt::DIST, mkt::COPY);
		//mkt::DArray<complex> output(0, 16, 16, complex{}, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<float> coeff(0, 16, 16, 0.0f, 1, 0, 0, mkt::DIST, mkt::COPY);
		
		Init_map_in_place_array_functor init_map_in_place_array_functor{};
		Zero_fill_map_in_place_array_functor zero_fill_map_in_place_array_functor{};
		Bitrev_reorder_map_index_in_place_array_functor bitrev_reorder_map_index_in_place_array_functor{input};
		Fetch_map_index_in_place_array_functor fetch_map_index_in_place_array_functor{input};
		Combine_map_index_in_place_array_functor combine_map_index_in_place_array_functor{input_double};
		
		
				
		
		int ntaps = 16;
		int nchans = 16;
		int nspectra = 16;
		mkt::map_in_place<float, Init_map_in_place_array_functor>(input, init_map_in_place_array_functor);
		mkt::map_in_place<float, Init_map_in_place_array_functor>(coeff, init_map_in_place_array_functor);
		mkt::map_in_place<float, Zero_fill_map_in_place_array_functor>(input_double, zero_fill_map_in_place_array_functor);
		mkt::sync_streams();
		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		int log2size = 4;
		bitrev_reorder_map_index_in_place_array_functor.log2size = (log2size);
		mkt::map_index_in_place<float, Bitrev_reorder_map_index_in_place_array_functor>(input_double, bitrev_reorder_map_index_in_place_array_functor);
		for(int i = 0; ((i) < 1); i++){
			for(int j = 0; ((j) < (log2size)); j++){
				fetch_map_index_in_place_array_functor.counter = (j);fetch_map_index_in_place_array_functor.log2size = (log2size);
				mkt::map_index_in_place<float, Fetch_map_index_in_place_array_functor>(input_double, fetch_map_index_in_place_array_functor);
				combine_map_index_in_place_array_functor.counter = (j);combine_map_index_in_place_array_functor.log2size = (log2size);combine_map_index_in_place_array_functor.pi = (PI);combine_map_index_in_place_array_functor.Problemsize = 16;
				mkt::map_index_in_place<float, Combine_map_index_in_place_array_functor>(input, combine_map_index_in_place_array_functor);
			}
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
