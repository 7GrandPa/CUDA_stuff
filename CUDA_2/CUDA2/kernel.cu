
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <iostream>

cudaError_t cudaMalloc(void** devPTr, size_t size);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);

__global__ void add(float *x, float *y, float *z, int n)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) {
		z[i] = x[i] + y[i];
	}   
}

int main()
{
	int N = 1 << 20;
	int nBytes = N * sizeof(float);

	// apply for host memory
	float *x, *y, *z;
	x = (float*)malloc(nBytes);
	y = (float*)malloc(nBytes);
	z = (float*)malloc(nBytes);

	// init the data
	for (int i=0; i < N; i++) {
		x[i] = 10.0;
		y[i] = 20.0;
	}

	// apply for device mem
	float *d_x, *d_y, *d_z;
	cudaMalloc((void**)&d_x, nBytes);
	cudaMalloc((void**)&d_y, nBytes);
	cudaMalloc((void**)&d_z, nBytes);

	// copy data from host to device
	cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)d_y, (void*)y, nBytes, cudaMemcpyHostToDevice);

	// define the kernel configuration
	dim3 blockSize(256);
	dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

	//execute
	add << < gridSize, blockSize >> > (d_x, d_y, d_z, N);

	//copy data from gpu to host
	cudaMemcpy((void*)z, (void*)d_z, nBytes, cudaMemcpyHostToDevice);

	//
	float maxError = 0.0;
	for (int i = 0; i < N; i++)
		maxError = fmax(maxError, fabs(z[i] - 30.0));
	std::cout << "×î´óÎó²î: " << maxError << std::endl;

	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);

	//free host mem
	free(x);
	free(y);
	free(z);

	return 0;
}


