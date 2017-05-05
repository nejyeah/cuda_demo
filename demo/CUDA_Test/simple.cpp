#include "simple.hpp"
#include <stdlib.h>
#include <iostream>
#include "common.hpp"

// =========================== vector add =============================
int test_vectorAdd()
{
	// Vector addition: C = A + B, implements element by element vector addition
	const int numElements{ 50000 };
	float* A = new float[numElements];
	float* B = new float[numElements];
	float* C1 = new float[numElements];
	float* C2 = new float[numElements];

	// Initialize vector
	for (int i = 0; i < numElements; ++i) {
		A[i] = rand() / (float)RAND_MAX;
		B[i] = rand() / (float)RAND_MAX;
	}

	int ret = vectorAdd_cpu(A, B, C1, numElements);
	if (ret != 0) PRINT_ERROR_INFO(vectorAdd_cpu);

	ret = vectorAdd_gpu(A, B, C2, numElements);
	if (ret != 0) PRINT_ERROR_INFO(vectorAdd_gpu);

	for (int i = 0; i < numElements; ++i) {
		if (fabs(C1[i] - C2[i]) > 1e-5) {
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			return -1;
		}
	}

	delete[] A;
	delete[] B;
	delete[] C1;
	delete[] C2;

	return 0;
}

int vectorAdd_cpu(const float *A, const float *B, float *C, int numElements)
{
	for (int i = 0; i < numElements; ++i) {
		C[i] = A[i] + B[i];
	}

	return 0;
}
