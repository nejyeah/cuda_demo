#include "simple.hpp"
#include <iostream>
#include <cuda_runtime.h> // For the CUDA runtime routines (prefixed with "cuda_")
#include <device_launch_parameters.h>

// reference: C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0\0_Simple

// =========================== vector add =============================
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements) {
		C[i] = A[i] + B[i];
	}
}

int vectorAdd_gpu(const float *A, const float *B, float *C, int numElements)
{
	// Error code to check return values for CUDA calls
	cudaError_t err{ cudaSuccess };
	size_t length{ numElements * sizeof(float) };
	fprintf(stderr, "Length: %d\n", length);
	float* d_A{ nullptr };
	float* d_B{ nullptr };
	float* d_C{ nullptr };

	err = cudaMalloc(&d_A, length);
	if (err != cudaSuccess) {
			fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
			return -1;
	}
	err = cudaMalloc(&d_B, length);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}
	err = cudaMalloc(&d_C, length);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}

	err = cudaMemcpy(d_A, A, length, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}
	err = cudaMemcpy(d_B, B, length, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}

	// Launch the Vector Add CUDA kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	fprintf(stderr, "CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	vectorAdd << <blocksPerGrid, threadsPerBlock >> >(d_A, d_B, d_C, numElements);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}

	// Copy the device result vector in device memory to the host result vector in host memory.
	err = cudaMemcpy(C, d_C, length, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}

	err = cudaFree(d_A);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}

	err = cudaFree(d_B);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}

	err = cudaFree(d_C);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}

	return err;
}
