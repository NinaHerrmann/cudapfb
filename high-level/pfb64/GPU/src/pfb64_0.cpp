	
	#include <omp.h>
	#include <openacc.h>
	#include <stdlib.h>
	#include <math.h>
	#include <array>
	#include <vector>
	#include <sstream>
	#include <chrono>
	#include <random>
	#include <limits>
	#include <memory>
	#include <cstddef>
	#include <type_traits>
	//#include <cuda.h>
	//#include <openacc_curand.h>
	
	#include "../include/musket.hpp"
	#include "../include/pfb64_0.hpp"
	
	
	std::vector<std::mt19937> random_engines;
	std::array<float*, 1> rns_pointers;
	std::array<float, 100000> rns;	
	std::vector<std::uniform_real_distribution<float>> rand_dist_float_0_0f_1_0f;
	
			
	const double PI = 3.141592653589793;
	mkt::DArray<float> input(0, 16, 16, 0.0f, 1, 0, 0, mkt::DIST, mkt::COPY);
	mkt::DArray<float> input_double(0, 16, 16, 0.0f, 1, 0, 0, mkt::DIST, mkt::COPY);
	mkt::DArray<complex> output(0, 16, 16, complex{}, 1, 0, 0, mkt::DIST, mkt::COPY);
	mkt::DArray<float> coeff(0, 16, 16, 0.0f, 1, 0, 0, mkt::DIST, mkt::COPY);
	
	//Complex::Complex() : x(), y() {}
	

	
	struct Init_map_in_place_array_functor{
		
		Init_map_in_place_array_functor(std::array<float*, 1> rns_pointers){
			for(int gpu = 0; gpu < 1; gpu++){
			 	_rns_pointers[gpu] = rns_pointers[gpu];
			}
			_rns_index = 0;
		}
		
		~Init_map_in_place_array_functor() {}
		
		auto operator()(int x){
			size_t local_rns_index  = _gang + _worker + _vector + _rns_index; // this can probably be improved
			local_rns_index  = (local_rns_index + 0x7ed55d16) + (local_rns_index << 12);
			local_rns_index = (local_rns_index ^ 0xc761c23c) ^ (local_rns_index >> 19);
			local_rns_index = (local_rns_index + 0x165667b1) + (local_rns_index << 5);
			local_rns_index = (local_rns_index + 0xd3a2646c) ^ (local_rns_index << 9);
			local_rns_index = (local_rns_index + 0xfd7046c5) + (local_rns_index << 3);
			local_rns_index = (local_rns_index ^ 0xb55a4f09) ^ (local_rns_index >> 16);
			local_rns_index = local_rns_index % 100000;
			_rns_index++;
			return static_cast<float>(static_cast<float>(_rns[local_rns_index++] * (1.0f - 0.0f + 0.999999) + 0.0f));
		}
	
		void init(int gpu){
			_rns = _rns_pointers[gpu];
			std::random_device rd{};
			std::mt19937 d_rng_gen(rd());
			std::uniform_int_distribution<> d_rng_dis(0, 100000);
			_rns_index = d_rng_dis(d_rng_gen);
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		
		
		float* _rns;
		std::array<float*, 1> _rns_pointers;
		size_t _rns_index;
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct Zero_fill_map_in_place_array_functor{
		
		Zero_fill_map_in_place_array_functor(){
		}
		
		~Zero_fill_map_in_place_array_functor() {}
		
		auto operator()(int x){
			return static_cast<float>(0);
		}
	
		void init(int gpu){
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct Bitrev_reorder_map_index_in_place_array_functor{
		
		Bitrev_reorder_map_index_in_place_array_functor(const mkt::DArray<float>& _input) : input(_input){
		}
		
		~Bitrev_reorder_map_index_in_place_array_functor() {}
		
		auto operator()(int x, int i){
			int brev = 0;
			int __powf = 0;
			return // TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed
			;
		}
	
		void init(int gpu){
			input.init(gpu);
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		int log2size;
		
		mkt::DeviceArray<float> input;
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct Fetch_map_index_in_place_array_functor{
		
		Fetch_map_index_in_place_array_functor(const mkt::DArray<float>& _input) : input(_input){
		}
		
		~Fetch_map_index_in_place_array_functor() {}
		
		auto operator()(int i, float Ti){
			return // TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed
			;
		}
	
		void init(int gpu){
			input.init(gpu);
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		int counter;
		int log2size;
		
		mkt::DeviceArray<float> input;
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct Combine_map_index_in_place_array_functor{
		
		Combine_map_index_in_place_array_functor(const mkt::DArray<float>& _input_double) : input_double(_input_double){
		}
		
		~Combine_map_index_in_place_array_functor() {}
		
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
			 + ((temp) * (Ai)));
			}
			 else {
					newa = ((Ai) + ((temp) * // TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed
					));
				}
			return (newa);
		}
	
		void init(int gpu){
			input_double.init(gpu);
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		int counter;
		int log2size;
		double pi;
		int Problemsize;
		
		mkt::DeviceArray<float> input_double;
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	
	
	
	
	
	
	
	int main(int argc, char** argv) {
		
		
		random_engines.reserve(0);
		std::mt19937 d_rng_gen(rd());
		std::uniform_real_distribution<float> d_rng_dis(0.0f, 1.0f);
		for(int random_number = 0; random_number < 100000; random_number++){
			rns[random_number] = d_rng_dis(d_rng_gen);
		}
		
		#pragma omp parallel for
		for(int gpu = 0; gpu < 1; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			float* devptr = static_cast<float*>(acc_malloc(100000 * sizeof(float)));
			rns_pointers[gpu] = devptr;
			acc_memcpy_to_device(devptr, rns.data(), 100000 * sizeof(float));
		}
		
		Init_map_in_place_array_functor init_map_in_place_array_functor{rns_pointers};
		Zero_fill_map_in_place_array_functor zero_fill_map_in_place_array_functor{};
		Bitrev_reorder_map_index_in_place_array_functor bitrev_reorder_map_index_in_place_array_functor{input};
		Fetch_map_index_in_place_array_functor fetch_map_index_in_place_array_functor{input};
		Combine_map_index_in_place_array_functor combine_map_index_in_place_array_functor{input_double};
		
		rand_dist_float_0_0f_1_0f.reserve(0);
		for(size_t counter = 0; counter < 0; ++counter){
			rand_dist_float_0_0f_1_0f.push_back(std::uniform_real_distribution<float>(0.0f, 1.0f));
		}
		
				
		
		int ntaps = 16;
		int nchans = 16;
		int nspectra = 16;
		mkt::map_in_place<float, Init_map_in_place_array_functor>(input, init_map_in_place_array_functor);
		mkt::map_in_place<float, Init_map_in_place_array_functor>(coeff, init_map_in_place_array_functor);
		mkt::map_in_place<float, Zero_fill_map_in_place_array_functor>(input_double, zero_fill_map_in_place_array_functor);
		for(int gpu = 0; gpu < 1; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			acc_wait_all();
		}
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
		for(int gpu = 0; gpu < 1; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			acc_wait_all();
		}
		std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
		double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
		
		printf("Execution time: %.5fs\n", seconds);
		printf("Threads: %i\n", 0);
		printf("Processes: %i\n", 1);
		
		#pragma omp parallel for
		for(int gpu = 0; gpu < 1; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			acc_free(rns_pointers[gpu]);
		}
		return EXIT_SUCCESS;
		}
