#include "simple.hpp"
#include <random>
#include <iostream>
#include <vector>
#include "common.hpp"

// =========================== vector add =============================
int test_vectorAdd()
{
	// Vector addition: C = A + B, implements element by element vector addition
	const int numElements{ 50000 };
	std::vector<float> A(numElements), B(numElements), C1(numElements), C2(numElements);

	// Initialize vector
	for (int i = 0; i < numElements; ++i) {
		A[i] = rand() / (float)RAND_MAX;
		B[i] = rand() / (float)RAND_MAX;
	}

	int ret = vectorAdd_cpu(A.data(), B.data(), C1.data(), numElements);
	if (ret != 0) PRINT_ERROR_INFO(vectorAdd_cpu);

	ret = vectorAdd_gpu(A.data(), B.data(), C2.data(), numElements);
	if (ret != 0) PRINT_ERROR_INFO(vectorAdd_gpu);

	for (int i = 0; i < numElements; ++i) {
		if (fabs(C1[i] - C2[i]) > EXP) {
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			return -1;
		}
	}

	return 0;
}

int vectorAdd_cpu(const float* A, const float* B, float* C, int numElements)
{
	for (int i = 0; i < numElements; ++i) {
		C[i] = A[i] + B[i];
	}

	return 0;
}
