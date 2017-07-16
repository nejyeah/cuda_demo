#include "funset.hpp"
#include <random>
#include <iostream>
#include <vector>
#include "common.hpp"

int test_vector_add()
{
	// Vector addition: C = A + B, implements element by element vector addition
	const int numElements{ 50000 };
	std::vector<float> A(numElements), B(numElements), C1(numElements), C2(numElements);

	// Initialize vector
	for (int i = 0; i < numElements; ++i) {
		A[i] = rand() / (float)RAND_MAX;
		B[i] = rand() / (float)RAND_MAX;
	}

	float elapsed_time1{ 0.f }, elapsed_time2{ 0.f }; // milliseconds
	int ret = vector_add_cpu(A.data(), B.data(), C1.data(), numElements, &elapsed_time1);
	if (ret != 0) PRINT_ERROR_INFO(vectorAdd_cpu);

	ret = vector_add_gpu(A.data(), B.data(), C2.data(), numElements, &elapsed_time2);
	if (ret != 0) PRINT_ERROR_INFO(vectorAdd_gpu);

	for (int i = 0; i < numElements; ++i) {
		if (fabs(C1[i] - C2[i]) > EXP) {
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			return -1;
		}
	}

	fprintf(stderr, "cpu run time: %f ms, gpu run time: %f ms\n", elapsed_time1, elapsed_time2);

	return 0;
}

