
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <cufft.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <cstring>
#include <iosfwd>

#define M_PI       3.14159265358979323846   // pi

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Complex numbers operations
static __device__ __host__ inline float2 CplxAdd(float2 a, float2 b) {
	float2 c; c.x = a.x + b.x; c.y = a.y + b.y; return c;
}

static __device__ __host__ inline float2 CplxInv(float2 a) {
	float2 c; c.x = -a.x; c.y = -a.y; return c;
}

static __device__ __host__ inline float2 CplxMul(float2 a, float2 b) {
	float2 c; c.x = a.x * b.x - a.y + b.y; c.y = a.x * b.y + a.y * b.x; return c;
}

/**
 * Reorders array by bit-reversing the indexes.
 */
__global__ void bitrev_reorder(float2* __restrict__ r, float2* __restrict__ d, int s, size_t nthr) {
	int id = blockIdx.x * nthr + threadIdx.x;
	// if (threadIdx.x == 0) printf("take id %d write to %d\n", id, __brev(id) >> (32 - s));
	r[__brev(id) >> (32 - s)] = d[id];
}
/**
 * Inner part of FFT loop. Contains the procedure itself.
 */
__device__ void inplace_fft_inner(float2* __restrict__ r, int j, int k, int m, int n) {
	if (j + k + m / 2 < n) {
		float2 t, u;

		t.x = __cosf((2.0 * M_PI * k) / (1.0 * m));
		t.y = -__sinf((2.0 * M_PI * k) / (1.0 * m));

		u = r[j + k];
		t = CplxMul(t, r[j + k + m / 2]);

		r[j + k] = CplxAdd(u, t);
		r[j + k + m / 2] = CplxAdd(u, CplxInv(t));
	}
}
/**
 * Middle part of FFT for small scope paralelism.
 */
__global__ void inplace_fft(float2* __restrict__ r, int j, int m, int n, size_t nthr) {
	int k = blockIdx.x * nthr + threadIdx.x;
	inplace_fft_inner(r, j, k, m, n);
}

/**
 * Outer part of FFT for large scope paralelism.
 */
__global__ void inplace_fft_outer(float2* __restrict__ r, int m, int n, size_t nthr) {
	int j = (blockIdx.x * nthr + threadIdx.x) * m;

	for (int k = 0; k < m / 2; k++) {
		inplace_fft_inner(r, j, k, m, n);
	}
}
__device__ float2 complex_mult(float2 returnvalue, float2 Ai) {
	float2 result;
	result.x = (returnvalue.x * Ai.x - returnvalue.y * Ai.y);
	result.y = (returnvalue.x * Ai.y + returnvalue.y * Ai.x);
	return result;
}
__device__ float2 complex_add(float2 returnvalue, float2 Ai) {
	float2 result;
	result.x = returnvalue.x + Ai.x;
	result.y = returnvalue.y + Ai.y;
	return result;
}
__global__ void muesli_combine(float2* T, float2* R, int log2size, int j, int Problemsize) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int b = i >> (log2size - j - 1);   int b2 = 0;
	for (int l = 0; l <= j; l++) {
		b2 = (b & 1) ? 2 * b2 + 1 : 2 * b2;
		b >>= 1;
	}
	float2 Ai = T[i];
	double v = 2.0 * M_PI / Problemsize * (b2 << (log2size - j - 1));
	float2 returnvalue;
	returnvalue.x = cos(v);
	returnvalue.y = sin(v);
	//inline complex combine(const DistributedArray<complex> & T, int j, int i, complex Ai) {
	float2 res;
	(i & (1 << log2size - 1 - j)) ? res = complex_add(T[i], complex_mult(returnvalue, Ai)) : res = complex_add(Ai, complex_mult(returnvalue, T[i]));
	R[i] = res;
	//}
}
__global__ void muesli_fetch(float2* T, float2* R, int log2size, int j) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	//fetch(const DistributedArray<complex> & R, int j, int i, complex Ti) {
	//return R.get(bitcomplement(log2size - 1 - j, i));
	T[i] = R[dev_bitcomplement(log2size - 1 - j, i)];
}

__global__ void bitcomplement(float2* result, float2* temp) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	// TODO return i ^ (1<<k);
}

__device__ int dev_bitcomplement(int k, int i) {
	return i ^ (1 << k);
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };
	// Copy data to GPU
	float2* r;
	float2* dn;
	float2* h_dn;
	int n = 1024;
	size_t data_size = n * sizeof(float2);
	h_dn = (float2*)malloc(data_size);
	cudaMalloc((void**)& r, data_size);
	cudaMalloc((void**)& dn, data_size);
	for (int i = 0; i < n; i++) {
		h_dn[i].x = 4;
		h_dn[i].y = 4;
		//printf("(%f,%f)", h_dn[i].x, h_dn[i].y);
	}
	cudaMemcpy(dn, h_dn, data_size, cudaMemcpyHostToDevice);
	//cudaMemcpy(dn, d, data_size, cudaMemcpyHostToDevice);
    // Add vectors in parallel.
	int threads = 32;
	int log2size = log2(n);
	dim3 dim_blocks(n / threads);
	dim3 dim_threads(threads);
	bitrev_reorder << <dim_blocks, dim_threads >> > (r, dn, log2size, threads);
	cudaMemcpy(h_dn, r, data_size, cudaMemcpyDeviceToHost);
	printf("%d blocks %d %d %d\n", ceil(n / threads), n, threads, n / threads);
	for (int i = 0; i < 10; i++) {
		printf("(%f,%f)\n", h_dn[i].x, h_dn[i].y);
	}
	float2* im_result;
	im_result = (float2*)malloc(data_size);
	// log2size = 9
	cudaDeviceSynchronize();
	// Iterative FFT (with loop paralelism balancing)
	/*for (int i = 1; i <= log2size; i++) {
		int m = 1 << i;
		if (n / m > 0) {
			printf("n / m / threads %d %d %d %d ceil:%d\n", n / m / threads, n, m, threads, ceil(n / m / threads));
			inplace_fft_outer << <((float)n / m / threads), threads >> > (r, m, n, threads);
			cudaMemcpy(im_result, r, data_size, cudaMemcpyDeviceToHost);
			for (int i = 0; i < 10; i++) {
				//printf("(%f,%f),", im_result[i].x, im_result[i].y);
			}
			//printf("\n");
		}
		else {
			for (int j = 0; j < n; j += m) {
				float repeats = m / 2;
				inplace_fft << <ceil(repeats / threads), threads >> > (r, j, m, n, threads);
				printf("balance\n");
			}
		}
	}*/
	r = dn;
	// move everything
	bitcomplement<< <((float)n / m / threads), threads >> > (dn, r);
	// T.permutePartition(curry(bitcomplement)(log2p - 1 - j));
	// R.mapIndexInPlace(curry(combine)(T)(j)); --> calculate individuals
	int Problemsize = data_size;
	int j = 0;
	muesli_combine << <((float)n / m / threads), threads >> > (dn, r, log2size, j, Problemsize);
	
	for (int j = 0; j < log2size; j++) {
		// T.mapIndexInPlace(curry(fetch)(R)(j));
		// (float2* T, float2* R, int log2size, int j)
		muesli_fetch << <((float)n / m / threads), threads >> > (dn, r, log2size, j);
		muesli_combine << <((float)n / m / threads), threads >> > (dn, r, log2size, j, Problemsize);
	}
	float2* result;
	result = (float2*)malloc(data_size);
	cudaMemcpy(result, r, data_size, cudaMemcpyDeviceToHost);
	for (int i = 0; i < 1024 / 2; i++) {
		//printf("(%f,%f),", result[i].x, result[i].y);
	}
	cudaFree(r);
	cudaFree(dn);
	// Making a cufft comparison...
	cufftHandle plan;
	cufftComplex* data;
	cufftComplex* cufft_output;
	float2* cufft_h_dn;
	cufft_h_dn = (float2*)malloc(data_size);
	cudaMalloc((void**)& data, sizeof(cufftComplex) * 1024);
	cudaMalloc((void**)& cufft_output, sizeof(cufftComplex) * 1024);
	for (int i = 0; i < 1024; i++) {
		cufft_h_dn[i].x = 4;
		cufft_h_dn[i].y = 4;
		//printf("(%f,%f)", h_dn[i].x, h_dn[i].y);
	}
	cudaMemcpy(data, cufft_h_dn, data_size, cudaMemcpyHostToDevice);
	cufftPlan1d(&plan, 32, CUFFT_C2C, 1024);
	cudaError_t cudaStatus = cudaGetLastError();

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cufft plan failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cufftExecC2C(plan, data, cufft_output, CUFFT_FORWARD);
	
	cudaDeviceSynchronize();
	cudaMemcpy(cufft_h_dn, cufft_output, data_size, cudaMemcpyDeviceToHost);
	for (int i = 0; i < 10; i++) {
		(cufft_h_dn[i].x == result[i].x) ? printf("same: %f==%f;\n", cufft_h_dn[i].x, result[i].x) : printf("%f!=%f; %f!=%f; \n", cufft_h_dn[i].x, result[i].x, cufft_h_dn[i].y, result[i].y);
	}

	cufftDestroy(plan);
	cudaFree(data);

    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n", c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
